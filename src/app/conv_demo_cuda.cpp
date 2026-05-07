#include "conv_demo_cuda.hpp"
#include "main.hpp"
#include <iostream>


#if !NEURAL_CUDA_ENABLED || !NEURAL_CUDNN_ENABLED

void run_conv_demo_cuda()
{
	std::cout << "conv_demo_cuda: build with -DNEURAL_ENABLE_CUDA=ON (cuDNN required).\n";
}

#else

#include "cifar10_loader.hpp"
#include "image_processing/image_augmentation.hpp"
#include "metrics.hpp"
#include "convolutional_box_static.hpp"
#include "convolutional_layer.hpp"
#include "cross_entropy_softmax_loss.hpp"
#include "cuda_tensor.hpp"
#include "cuda_tensor_n.hpp"
#include "linear_layer.hpp"
#include "max_pool_layer.hpp"
#include "neural_cuda_runtime.hpp"
#include "neural_cuda_layer_sync.hpp"
#include <cuda_runtime.h>
#include "relu_layer.hpp"
#include "batch_norm_1d_layer.hpp"
#include "dropout_layer.hpp"
#include "momentum_sgd_optimizer.hpp"
#include "cosine_scheduling.hpp"
#include "training_epoch_progress_reporter.hpp"
#include "sequential_nn_static.hpp"
#include "tensor.hpp"
#include <omp.h>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <random>
#include <memory>
#include <iomanip>

namespace {

void cuda_sync_checked( char const *context )
{
	cudaError_t const e = cudaDeviceSynchronize();
	if ( e != cudaSuccess ) {
		std::cout << "conv_demo_cuda: cudaDeviceSynchronize after " << context << ": "
		          << cudaGetErrorString( e ) << "\n";
	}
}

using Tensor2D_t = neural::CudaTensor<float>;
using TensorN_t  = neural::CudaTensor4<float>;

// Exact layer order from https://github.com/geifmany/cifar-vgg/blob/master/cifar10vgg.py
// \c build_model(): Conv→ReLU→BN, Keras \c Dropout(p) → \c DropoutLayer(1-p) (inverted dropout).
using ConvBox_t = neural::ConvolutionalBox_static<
    TensorN_t, Tensor2D_t,
    neural::ConvolutionalLayer<TensorN_t>, neural::ReLULayer<TensorN_t>,
    neural::BatchNormLayer<TensorN_t>, neural::DropoutLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::ReLULayer<TensorN_t>,
    neural::BatchNormLayer<TensorN_t>, neural::MaxPoolLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::ReLULayer<TensorN_t>,
    neural::BatchNormLayer<TensorN_t>, neural::DropoutLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::ReLULayer<TensorN_t>,
    neural::BatchNormLayer<TensorN_t>, neural::MaxPoolLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::ReLULayer<TensorN_t>,
    neural::BatchNormLayer<TensorN_t>, neural::DropoutLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::ReLULayer<TensorN_t>,
    neural::BatchNormLayer<TensorN_t>, neural::DropoutLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::ReLULayer<TensorN_t>,
    neural::BatchNormLayer<TensorN_t>, neural::MaxPoolLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::ReLULayer<TensorN_t>,
    neural::BatchNormLayer<TensorN_t>, neural::DropoutLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::ReLULayer<TensorN_t>,
    neural::BatchNormLayer<TensorN_t>, neural::DropoutLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::ReLULayer<TensorN_t>,
    neural::BatchNormLayer<TensorN_t>, neural::MaxPoolLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::ReLULayer<TensorN_t>,
    neural::BatchNormLayer<TensorN_t>, neural::DropoutLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::ReLULayer<TensorN_t>,
    neural::BatchNormLayer<TensorN_t>, neural::DropoutLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::ReLULayer<TensorN_t>,
    neural::BatchNormLayer<TensorN_t>, neural::MaxPoolLayer<TensorN_t>,
    neural::DropoutLayer<TensorN_t>>;

using ConvNN_t = neural::SequentialNN_static<
    Tensor2D_t,
    neural::SoftmaxCrossEntropyLoss<Tensor2D_t>,
    ConvBox_t,
    neural::LinearLayer<Tensor2D_t>,
    neural::ReLULayer<Tensor2D_t>,
    neural::BatchNorm1dLayer<Tensor2D_t>,
    neural::DropoutLayer<Tensor2D_t>,
    neural::LinearLayer<Tensor2D_t>>;

/// Batch norm + dropout train/eval: \p training true during SGD, false for eval.
void set_train_mode( ConvNN_t &nn, bool training )
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
float mean_cross_entropy_on_host( ConvNN_t &nn, std::vector<neural::Tensor<float>> const &images,
                                  std::vector<neural::Tensor<float>> const &labels,
                                  std::size_t batch_size )
{
	if ( images.empty() ) {
		return 0.f;
	}
	neural::SoftmaxCrossEntropyLoss<neural::Tensor<float>> loss_fn;
	double sum_weighted = 0.0;
	std::size_t const n         = images.size();
	std::size_t const in_cols   = images[0].cols();
	std::size_t const out_cols  = labels[0].cols();
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

} // namespace

void run_conv_demo_cuda()
{
	neural::MainSignal::instance().reset_interrupt_flag();
	neural::MainSignal::instance().install_sigint_handler();
	std::cout << "Conv demo (CIFAR-10, CUDA)\n";

	if ( !neural::cuda_runtime_ready() ) {
		std::cout << "No CUDA device available.\n";
		return;
	}
	cuda_sync_checked( "cuda_runtime_ready" );

	std::filesystem::path const bin_dir = neural::find_cifar10_bin_dir();
	if ( bin_dir.empty() ) {
		std::cout << "Could not find CIFAR-10 binaries. Download cifar-10-binary.tar.gz from\n"
		             "  https://www.cs.toronto.edu/~kriz/cifar.html\n"
		             "and extract so that data_batch_1.bin is under data/cifar/cifar-10-batches-bin/\n";
		return;
	}

	// All five `data_batch_*.bin` files → 50 000 training samples. The official
	// `test_batch.bin` (10 000) is used for both periodic val (if enabled) and final test.
	auto train_data = neural::load_cifar10_train<neural::Tensor<float>>( bin_dir, 0 );
	if ( train_data.count == 0 ) {
		std::cout << "Failed to load training data from " << bin_dir.string() << "\n";
		return;
	}

	auto test_data = neural::load_cifar10_test<neural::Tensor<float>>( bin_dir, 0 );

	std::cout << "Loaded " << train_data.count << " training samples (host Tensor<float>)";
	if ( test_data.count > 0 )
		std::cout << ", " << test_data.count << " test samples (host) for val/test";
	std::cout << ".\n";
	std::cout << "Training: aug on [0,1] (flip+affine), then scalar normalize from train stats (test stored normalized).\n";
	auto const [norm_mean, norm_std] = neural::cifar10_compute_normalization( train_data.images );
	std::cout << "CIFAR-10 train statistics: mean=" << norm_mean << "  std=" << norm_std << "\n";
	if ( test_data.count > 0 )
		neural::cifar10_apply_normalization( test_data.images, norm_mean, norm_std );

	neural::CosineScheduling lr_schedule( 0.01f, 1e-4f, 1000 );
	std::cout << "LR schedule: cosine (max=" << lr_schedule.lr_max() << " → min=" << lr_schedule.lr_min()
	          << " over " << lr_schedule.cosine_epochs() << " epochs).\n";

	// Keras Dropout(0.3)/(0.4)/(0.5) → keep_prob 0.7/0.6/0.5
	float constexpr kDropKeras03 = 1.f - 0.3f;
	float constexpr kDropKeras04 = 1.f - 0.4f;
	float constexpr kDropKeras05 = 1.f - 0.5f;

	std::mt19937 rng( std::random_device{}() );
	std::uniform_int_distribution<std::uint32_t> rng_seed_dist( 0, std::numeric_limits<std::uint32_t>::max() );

	ConvBox_t box(
	    3, 32, 32,
	    neural::ConvolutionalLayer<TensorN_t>( 64, 3 ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::BatchNormLayer<TensorN_t>( 64, 1e-5f, 0.01f ),
	    neural::DropoutLayer<TensorN_t>( kDropKeras03, rng_seed_dist( rng ) ),
	    neural::ConvolutionalLayer<TensorN_t>( 64, 3 ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::BatchNormLayer<TensorN_t>( 64, 1e-5f, 0.01f ),
	    neural::MaxPoolLayer<TensorN_t>( 2, 2 ),
	    neural::ConvolutionalLayer<TensorN_t>( 128, 3 ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::BatchNormLayer<TensorN_t>( 128, 1e-5f, 0.01f ),
	    neural::DropoutLayer<TensorN_t>( kDropKeras04, rng_seed_dist( rng ) ),
	    neural::ConvolutionalLayer<TensorN_t>( 128, 3 ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::BatchNormLayer<TensorN_t>( 128, 1e-5f, 0.01f ),
	    neural::MaxPoolLayer<TensorN_t>( 2, 2 ),
	    neural::ConvolutionalLayer<TensorN_t>( 256, 3 ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::BatchNormLayer<TensorN_t>( 256, 1e-5f, 0.01f ),
	    neural::DropoutLayer<TensorN_t>( kDropKeras05, rng_seed_dist( rng ) ),
	    neural::ConvolutionalLayer<TensorN_t>( 256, 3 ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::BatchNormLayer<TensorN_t>( 256, 1e-5f, 0.01f ),
	    neural::DropoutLayer<TensorN_t>( kDropKeras05, rng_seed_dist( rng ) ),
	    neural::ConvolutionalLayer<TensorN_t>( 256, 3 ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::BatchNormLayer<TensorN_t>( 256, 1e-5f, 0.01f ),
	    neural::MaxPoolLayer<TensorN_t>( 2, 2 ),
	    neural::ConvolutionalLayer<TensorN_t>( 512, 3 ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::BatchNormLayer<TensorN_t>( 512, 1e-5f, 0.01f ),
	    neural::DropoutLayer<TensorN_t>( kDropKeras05, rng_seed_dist( rng ) ),
	    neural::ConvolutionalLayer<TensorN_t>( 512, 3 ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::BatchNormLayer<TensorN_t>( 512, 1e-5f, 0.01f ),
	    neural::DropoutLayer<TensorN_t>( kDropKeras05, rng_seed_dist( rng ) ),
	    neural::ConvolutionalLayer<TensorN_t>( 512, 3 ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::BatchNormLayer<TensorN_t>( 512, 1e-5f, 0.01f ),
	    neural::MaxPoolLayer<TensorN_t>( 2, 2 ),
	    neural::ConvolutionalLayer<TensorN_t>( 512, 3 ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::BatchNormLayer<TensorN_t>( 512, 1e-5f, 0.01f ),
	    neural::DropoutLayer<TensorN_t>( kDropKeras05, rng_seed_dist( rng ) ),
	    neural::ConvolutionalLayer<TensorN_t>( 512, 3 ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::BatchNormLayer<TensorN_t>( 512, 1e-5f, 0.01f ),
	    neural::DropoutLayer<TensorN_t>( kDropKeras05, rng_seed_dist( rng ) ),
	    neural::ConvolutionalLayer<TensorN_t>( 512, 3 ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::BatchNormLayer<TensorN_t>( 512, 1e-5f, 0.01f ),
	    neural::MaxPoolLayer<TensorN_t>( 2, 2 ),
	    neural::DropoutLayer<TensorN_t>( kDropKeras05, rng_seed_dist( rng ) ) );

	ConvNN_t nn( box,
	             neural::LinearLayer<Tensor2D_t>( 512 ), neural::ReLULayer<Tensor2D_t>(),
	             neural::BatchNorm1dLayer<Tensor2D_t>( 512, 1e-5f, 0.01f ),
	             neural::DropoutLayer<Tensor2D_t>( kDropKeras05, rng_seed_dist( rng ) ),
	             neural::LinearLayer<Tensor2D_t>( 10 ) );

	neural::MomentumSGDOptimizer<Tensor2D_t, ConvNN_t> opt( nn );
	opt.m_learning_rate = lr_schedule.lr_max();
	opt.m_momentum       = 0.9f;
	opt.m_batch_size     = 128;
	opt.m_epochs         = 250;
	opt.m_l2_regularizer = 0.0001f;
	opt.m_nesterov       = true;
	cuda_sync_checked( "model_and_optimizer_init" );

	std::vector<std::unique_ptr<neural::ImageAugmentation>> cifar_aug;
	cifar_aug.reserve( omp_get_max_threads() );
	for ( std::size_t i = 0; i < omp_get_max_threads(); ++i ) {
		cifar_aug.emplace_back( neural::make_cifar10_training_augmentation() );
	}
	neural::Cifar10AugmentBatchOp const cifar_batch_op(
	    train_data.images[0].cols(), train_data.labels[0].cols(), opt.m_batch_size, std::move( cifar_aug ),
	    norm_mean, norm_std );

	set_train_mode( nn, true );
	neural::TrainingEpochProgressReporter epoch_progress;
	opt.train(
	    train_data.images, train_data.labels,
	    [&]( std::size_t epoch, std::size_t epoch_max, std::size_t batch_idx,
	         std::size_t batch_max, float loss ) -> bool {
		    bool ans = false;
		    float const lr = lr_schedule.learning_rate_for_epoch( epoch );
		    if ( !epoch_progress.consume_batch( loss, batch_idx, batch_max, epoch, epoch_max, lr ) )
			    return ans;
		    opt.m_learning_rate = lr;

		    bool const signal_status = neural::MainSignal::instance().interrupt_requested();
		    return signal_status ? true : ans;
	    } );
	std::cout << "\n";
	cuda_sync_checked( "opt.train" );

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
