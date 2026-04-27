#include "conv_demo_cuda_small.hpp"
#include <iostream>
#include <csignal>

#if !NEURAL_CUDA_ENABLED || !NEURAL_CUDNN_ENABLED

void run_conv_demo_cuda_small()
{
	std::cout << "conv_demo_cuda_small: build with -DNEURAL_ENABLE_CUDA=ON (cuDNN required).\n";
}

#else

#include "cifar10_loader.hpp"
#include "image_processing/image_augmentation.hpp"
#include "metrics.hpp"
#include "convolutional_box.hpp"
#include "convolutional_layer.hpp"
#include "cross_entropy_softmax_loss.hpp"
#include "cuda_tensor.hpp"
#include "cuda_tensor_n.hpp"
#include "linear_layer.hpp"
#include "max_pool_layer.hpp"
#include "neural_cuda_runtime.hpp"
#include "neural_cuda_layer_sync.hpp"
#include "relu_layer.hpp"
#include "batch_norm_1d_layer.hpp"
#include "dropout_layer.hpp"
#include "momentum_sgd_optimizer.hpp"
#include "sequential_nn.hpp"
#include "tensor.hpp"
#include <omp.h>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <iomanip>
#include <random>
#include <string>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>

namespace {

using Tensor2D_t = neural::CudaTensor<float>;
using TensorN_t  = neural::CudaTensor4<float>;

// keep_prob = 1 disables dropout (inverted dropout identity in train).
float const kDropoutAfterBlock1 = 1.f;
float const kHeadDrop0           = 1.f;
float const kHeadDrop1           = 1.f;

// Architecture: 32×32 → pool → 16×16 → pool → 8×8 → pool → 4×4; channels 32 → 64 → 64;
// extra 3×3 conv + ReLU at 4×4 before head dropout.
// Flat: 4×4×64 = 1024. FC: 256 → 10.
using SmallConvBox_t = neural::ConvolutionalBox<
    TensorN_t, Tensor2D_t,
    neural::ConvolutionalLayer<TensorN_t>, neural::BatchNormLayer<TensorN_t>,
    neural::ReLULayer<TensorN_t>, neural::MaxPoolLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::BatchNormLayer<TensorN_t>,
    neural::ReLULayer<TensorN_t>, neural::DropoutLayer<TensorN_t>, neural::MaxPoolLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::BatchNormLayer<TensorN_t>,
    neural::ReLULayer<TensorN_t>, neural::MaxPoolLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::ReLULayer<TensorN_t>,
    neural::DropoutLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::BatchNormLayer<TensorN_t>,
    neural::ReLULayer<TensorN_t>, neural::MaxPoolLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::ReLULayer<TensorN_t>,
    neural::DropoutLayer<TensorN_t>
		>;

using SmallConvNN_t = neural::SequentialNN<
    Tensor2D_t,
    neural::SoftmaxCrossEntropyLoss<Tensor2D_t>,
    SmallConvBox_t,
    neural::LinearLayer<Tensor2D_t>,
    neural::BatchNorm1dLayer<Tensor2D_t>,
    neural::ReLULayer<Tensor2D_t>,
    neural::DropoutLayer<Tensor2D_t>,
    neural::LinearLayer<Tensor2D_t>>;

void set_train_mode( SmallConvNN_t &nn, bool training )
{
	nn.forEachLayer( [training]( auto &layer ) {
		if constexpr ( requires { layer.setTraining( training ); } ) {
			layer.setTraining( training );
		}
	} );
}

/// Mean cross-entropy on a host dataset (e.g. normalized test fold).
/// Caller must set \ref set_train_mode( \p nn, false ) then \ref nn.ensureBuffersForShape( batch_size, in_cols )
/// before calling. Restores \p nn buffer rows to \p batch_size before return (after any smaller tail batch).
float mean_cross_entropy_on_host( SmallConvNN_t &nn,
                                  std::vector<neural::Tensor<float>> const &images,
                                  std::vector<neural::Tensor<float>> const &labels,
                                  std::size_t batch_size )
{
	if ( images.empty() ) {
		return 0.f;
	}
	neural::SoftmaxCrossEntropyLoss<neural::Tensor<float>> loss_fn;
	double sum_weighted = 0.0;
	std::size_t const n = images.size();
	std::size_t const in_cols  = images[0].cols();
	std::size_t const out_cols = labels[0].cols();
	for ( std::size_t start = 0; start < n; start += batch_size ) {
		std::size_t const b = std::min( batch_size, n - start );
		nn.ensureBuffersForShape( b, in_cols );
		neural::Tensor<float> host_x( b, in_cols );
		neural::Tensor<float> host_y( b, out_cols );
		for ( std::size_t r = 0; r < b; ++r ) {
			host_x.assignTensorAsRow( r, images[start + r] );
			host_y.assignTensorAsRow( r, labels[start + r] );
		}
		Tensor2D_t dev_in( b, in_cols );
		dev_in.assign( host_x.data(), host_x.size() );
		Tensor2D_t logits_dev = nn.forward( dev_in );
		neural::Tensor<float> logits( logits_dev.rows(), logits_dev.cols() );
		logits_dev.copyToHost( logits.data() );
		neural::cuda_layer_sync();
		neural::Tensor<float> const loss_t = loss_fn.forward( logits, host_y );
		sum_weighted += static_cast<double>( loss_t( 0, 0 ) ) * static_cast<double>( b );
	}
	nn.ensureBuffersForShape( batch_size, in_cols );
	return static_cast<float>( sum_weighted / static_cast<double>( n ) );
}

/// Train accuracy with the same aug + shuffle recipe as \c MomentumSGDOptimizer::train for \p epoch index.
float train_accuracy_augmented_replay( std::size_t epoch, std::size_t batch_size,
                                       std::vector<neural::Tensor<float>> const &inputs,
                                       std::vector<neural::Tensor<float>> const &targets,
                                       neural::Cifar10AugmentBatchOp const &op, SmallConvNN_t &nn )
{
	std::size_t const n = inputs.size();
	std::size_t const total_batches = n / batch_size;
	if ( total_batches == 0 ) {
		return 0.f;
	}
	std::vector<std::size_t> batch_indices( n );
	std::iota( batch_indices.begin(), batch_indices.end(), 0 );
	std::shuffle( batch_indices.begin(), batch_indices.end(),
	              std::mt19937( static_cast<std::mt19937::result_type>( epoch ) ) );
	std::vector<int> batch_indices_int( n );
	for ( std::size_t r = 0; r < n; ++r ) {
		batch_indices_int[r] = static_cast<int>( batch_indices[r] );
	}
	std::size_t const in_cols = inputs[0].cols();
	neural::Tensor<float> host_x( batch_size, in_cols );
	neural::Tensor<float> host_y( batch_size, targets[0].cols() );
	set_train_mode( nn, true );
	nn.ensureBuffersForShape( batch_size, in_cols ); // always after train/eval toggle
	std::size_t correct = 0;
	for ( std::size_t k = 0; k < total_batches; ++k ) {
		std::size_t const window_start = k * batch_size;
		neural::assignBatch( host_x, host_y, inputs, targets, batch_indices_int, window_start,
		                     batch_size, host_x, host_y, op );
		Tensor2D_t dev_in( batch_size, in_cols );
		dev_in.assign( host_x.data(), host_x.size() );
		Tensor2D_t logits_dev = nn.forward( dev_in );
		neural::Tensor<float> logits( logits_dev.rows(), logits_dev.cols() );
		logits_dev.copyToHost( logits.data() );
		neural::cuda_layer_sync();
		auto const pred_idxs = logits.argmaxAlongAxis( 1 );
		for ( std::size_t i = 0; i < batch_size; ++i ) {
			std::size_t const pred = static_cast<std::size_t>( pred_idxs( i, 0 ) );
			std::size_t gt       = 0;
			for ( std::size_t c = 0; c < 10; ++c ) {
				if ( host_y( i, c ) > 0.5f ) {
					gt = c;
					break;
				}
			}
			if ( pred == gt ) {
				++correct;
			}
		}
	}
	return static_cast<float>( correct ) / static_cast<float>( total_batches * batch_size );
}

} // namespace

namespace {
std::sig_atomic_t volatile gSignalStatusSmall;
}

void signal_handler_small( int signal )
{
	gSignalStatusSmall = signal;
}

void run_conv_demo_cuda_small()
{
	std::signal( SIGINT, signal_handler_small );
	std::cout << "CIFAR-10 small conv demo (CUDA) — for dropout / pipeline debugging\n";

	if ( !neural::cuda_runtime_ready() ) {
		std::cout << "No CUDA device available.\n";
		return;
	}

	std::filesystem::path const bin_dir = neural::find_cifar10_bin_dir();
	if ( bin_dir.empty() ) {
		std::cout << "Could not find CIFAR-10 binaries.\n";
		return;
	}

	auto train_data = neural::load_cifar10_train<neural::Tensor<float>>( bin_dir, 0 );
	if ( train_data.count == 0 ) {
		std::cout << "Failed to load training data from " << bin_dir.string() << "\n";
		return;
	}
	auto test_data = neural::load_cifar10_test<neural::Tensor<float>>( bin_dir, 0 );

	std::cout << "Loaded " << train_data.count << " training samples";
	if ( test_data.count > 0 )
		std::cout << ", " << test_data.count << " test samples";
	std::cout << ".\n";

	auto const [norm_mean, norm_std] = neural::cifar10_compute_normalization( train_data.images );
	std::cout << "Train stats: mean=" << norm_mean << "  std=" << norm_std
	          << "  (aug on [0,1], normalize in batch op; test stored normalized)\n";
	if ( test_data.count > 0 )
		neural::cifar10_apply_normalization( test_data.images, norm_mean, norm_std );

	/// Same two schedules as `conv_demo_cuda.cpp`, scaled to this demo’s 0.01 base (vs 0.1 on the big VGG run).
	enum class LrScheduleKind : std::uint8_t { Cosine, Step };
	LrScheduleKind const kLrSchedule = LrScheduleKind::Step;
	float const       kLrMaxCosine       = 0.1f;
	float const       kLrMinCosine       = 1e-4f;
	std::size_t const kCosineLrOverEpochs = 250;
	float const       kLrBaseStep        = 0.1f;
	// Use 0.01 like `conv_demo_cuda.cpp` — 0.001 makes the first few epochs so slow that
	// the printed epoch-mean loss barely moves; this is a schedule choice, not a test failure.
	float const       kLrWarmupStart     = 0.01f;
	std::size_t const kWarmupEpochs      = 5;
	std::size_t const kLrDropEveryEpochs = 20;
	if ( kLrSchedule == LrScheduleKind::Cosine ) {
		std::cout << "LR schedule: cosine (max=" << kLrMaxCosine << " → min=" << kLrMinCosine
		          << " over " << kCosineLrOverEpochs << " epochs).\n";
	} else {
		std::cout << "LR schedule: step (base " << kLrBaseStep << " x 0.5^((epoch-warmup)/"
		          << kLrDropEveryEpochs << ")), warmup " << kWarmupEpochs << " epochs ("
		          << kLrWarmupStart << "→" << kLrBaseStep << ").\n";
	}

	std::mt19937 rng( std::random_device{}() );
	std::uniform_int_distribution<std::uint32_t> rng_seed_dist( 0, std::numeric_limits<std::uint32_t>::max() );

	// 3 / 2 / 2 / 2 stride-2 pools: 32 → 16 → 8 → 4; flat 4×4×64 = 1024.
	SmallConvBox_t box(
	    3, 32, 32,
	    neural::ConvolutionalLayer<TensorN_t>( 32, 3 ),
	    neural::BatchNormLayer<TensorN_t>( 32, 1e-5f, 0.01f ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::MaxPoolLayer<TensorN_t>( 2, 2 ),
	    neural::ConvolutionalLayer<TensorN_t>( 64, 3 ),
	    neural::BatchNormLayer<TensorN_t>( 64, 1e-5f, 0.01f ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::DropoutLayer<TensorN_t>( kDropoutAfterBlock1, rng_seed_dist( rng ) ),
	    neural::MaxPoolLayer<TensorN_t>( 2, 2 ),
	    neural::ConvolutionalLayer<TensorN_t>( 64, 3 ),
	    neural::BatchNormLayer<TensorN_t>( 64, 1e-5f, 0.01f ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::MaxPoolLayer<TensorN_t>( 2, 2 ),
	    neural::ConvolutionalLayer<TensorN_t>( 64, 3 ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::DropoutLayer<TensorN_t>( kHeadDrop0, rng_seed_dist( rng ) ),
	    neural::ConvolutionalLayer<TensorN_t>( 128, 3 ),
	    neural::BatchNormLayer<TensorN_t>( 128, 1e-5f, 0.01f ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::MaxPoolLayer<TensorN_t>( 2, 2 ),
	    neural::ConvolutionalLayer<TensorN_t>( 128, 3 ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::DropoutLayer<TensorN_t>( kHeadDrop0, rng_seed_dist( rng ) ) );

	SmallConvNN_t nn(
	    box,
	    neural::LinearLayer<Tensor2D_t>( 256 ),
	    neural::BatchNorm1dLayer<Tensor2D_t>( 256, 1e-5f, 0.01f ),
	    neural::ReLULayer<Tensor2D_t>(),
	    neural::DropoutLayer<Tensor2D_t>( kHeadDrop1, rng_seed_dist( rng ) ),
	    neural::LinearLayer<Tensor2D_t>( 10 ) );

	neural::MomentumSGDOptimizer<Tensor2D_t, SmallConvNN_t> opt( nn );
	opt.m_learning_rate  = kLrSchedule == LrScheduleKind::Cosine ? kLrMaxCosine : kLrWarmupStart;
	opt.m_momentum      = 0.9f;
	opt.m_batch_size    = 512;
	opt.m_epochs        = 40;
	opt.m_l2_regularizer = 0.0001f;
	opt.m_nesterov      = true;

	auto last_epoch_end  = std::chrono::steady_clock::now();
	float loss_sum_epoch = 0.f;

	std::vector<std::unique_ptr<neural::ImageAugmentation>> cifar_aug;
	cifar_aug.reserve( omp_get_max_threads() );
	for ( std::size_t i = 0; i < omp_get_max_threads(); ++i ) {
		cifar_aug.emplace_back( neural::make_cifar10_training_augmentation() );
	}
	neural::Cifar10AugmentBatchOp const cifar_batch_op(
	    train_data.images[0].cols(), train_data.labels[0].cols(), opt.m_batch_size, std::move( cifar_aug ),
	    norm_mean, norm_std );

	set_train_mode( nn, true );
	opt.train(
	    train_data.images, train_data.labels,
	    [&]( std::size_t epoch, std::size_t epoch_max, std::size_t batch_idx,
	         std::size_t batch_max, float loss ) -> bool {
		    bool ans = false;
		    loss_sum_epoch += loss;
		    if ( batch_idx + 1 < batch_max )
			    return ans;
		    float const mean_loss = loss_sum_epoch / static_cast<float>( batch_max );
		    loss_sum_epoch        = 0.f;
		    auto const now        = std::chrono::steady_clock::now();
		    double const elapsed =
		        std::chrono::duration<double>( now - last_epoch_end ).count();
		    last_epoch_end        = now;
		    float const lr_print  = opt.m_learning_rate;

		    float const tr_acc = train_accuracy_augmented_replay(
		        epoch, opt.m_batch_size, train_data.images, train_data.labels, cifar_batch_op, nn );
		    float val_loss = 0.f;
		    float val_acc  = 0.f;
		    if ( test_data.count > 0 ) {
			    set_train_mode( nn, false );
			    nn.ensureBuffersForShape( opt.m_batch_size, train_data.images[0].cols() );
			    val_loss = mean_cross_entropy_on_host( nn, test_data.images, test_data.labels,
			                                            opt.m_batch_size );
			    val_acc = neural::compute_accuracy( nn, test_data.images, test_data.labels, 10,
			                                          opt.m_batch_size );
		    }

		    set_train_mode( nn, true );
		    nn.ensureBuffersForShape( opt.m_batch_size, train_data.images[0].cols() );

		    if ( kLrSchedule == LrScheduleKind::Cosine ) {
			    float const progress = std::min(
			        static_cast<float>( epoch ) / static_cast<float>( kCosineLrOverEpochs ),
			        1.f );
			    opt.m_learning_rate = kLrMinCosine +
			                          0.5f * ( kLrMaxCosine - kLrMinCosine ) *
			                              ( 1.f + std::cos( std::numbers::pi_v<float> * progress ) );
		    } else {
			    if ( epoch < kWarmupEpochs ) {
				    float const t = static_cast<float>( epoch + 1 ) / static_cast<float>( kWarmupEpochs );
				    opt.m_learning_rate =
				        kLrWarmupStart + t * ( kLrBaseStep - kLrWarmupStart );
			    } else {
				    std::size_t const vgg_step = ( epoch - kWarmupEpochs ) / kLrDropEveryEpochs;
				    opt.m_learning_rate = kLrBaseStep * static_cast<float>(
				        std::pow( 0.5, static_cast<double>( vgg_step ) ) );
			    }
		    }

		    std::cout << std::fixed << "Epoch " << ( epoch + 1 ) << "/" << epoch_max << "  loss: "
		              << std::setprecision( 4 ) << mean_loss << "  val_loss: " << val_loss
		              << "  acc: " << tr_acc << "  val_acc: " << val_acc << "  lr: "
		              << std::setprecision( 6 ) << lr_print << "  time_s: " << std::setprecision( 3 )
		              << elapsed << "\n"
		              << std::flush;

		    bool const stop = ( gSignalStatusSmall != 0 );
		    return stop ? true : ans;
	    },
	    cifar_batch_op );

	float te_loss = 0.f;
	float te_acc  = 0.f;
	if ( test_data.count > 0 ) {
		set_train_mode( nn, false );
		nn.ensureBuffersForShape( opt.m_batch_size, train_data.images[0].cols() );
		te_loss = mean_cross_entropy_on_host( nn, test_data.images, test_data.labels,
		                                      opt.m_batch_size );
		te_acc = neural::compute_accuracy( nn, test_data.images, test_data.labels, 10,
		                                   opt.m_batch_size );
	}
	std::cout << std::fixed << std::setprecision( 4 ) << "Test  loss: " << te_loss
	          << "  acc: " << te_acc << "  (device: CUDA)\n";
}

#endif
