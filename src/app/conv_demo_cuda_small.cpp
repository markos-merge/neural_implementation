#include "conv_demo_cuda_small.hpp"
#include "main.hpp"
#include <iostream>

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
#include "momentum_sgd_optimizer.hpp"
#include "cosine_scheduling.hpp"
#include "training_epoch_progress_reporter.hpp"
#include "sequential_nn.hpp"
#include "tensor.hpp"
#include <omp.h>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace {

using Tensor2D_t = neural::CudaTensor<float>;
using TensorN_t  = neural::CudaTensor4<float>;

// Small debug topology (BN-stabilized, hidden Linear head):
//   Conv(3→32)  + BN + ReLU + MaxPool(2,2)   // 32→16
//   Conv(32→64) + BN + ReLU + MaxPool(2,2)   // 16→8
//   Conv(64→128)+ BN + ReLU + MaxPool(2,2)   // 8→4
//   Flatten (4*4*128 = 2048) → Linear(2048→128) + BN1d + ReLU → Linear(128→10)
using SmallConvBox_t = neural::ConvolutionalBox<
    TensorN_t, Tensor2D_t,
    neural::ConvolutionalLayer<TensorN_t>, neural::BatchNormLayer<TensorN_t>,
    neural::ReLULayer<TensorN_t>, neural::MaxPoolLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::BatchNormLayer<TensorN_t>,
    neural::ReLULayer<TensorN_t>, neural::MaxPoolLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::BatchNormLayer<TensorN_t>,
    neural::ReLULayer<TensorN_t>, neural::MaxPoolLayer<TensorN_t>>;

using SmallConvNN_t = neural::SequentialNN<
    Tensor2D_t,
    neural::SoftmaxCrossEntropyLoss<Tensor2D_t>,
    SmallConvBox_t,
    neural::LinearLayer<Tensor2D_t>,
    neural::BatchNorm1dLayer<Tensor2D_t>,
    neural::ReLULayer<Tensor2D_t>,
    neural::LinearLayer<Tensor2D_t>>;

void set_train_mode( SmallConvNN_t &nn, bool training )
{
	nn.forEachLayer( [training]( auto &layer ) {
		if constexpr ( requires { layer.setTraining( training ); } ) {
			layer.setTraining( training );
		}
	} );
}
} // namespace

void run_conv_demo_cuda_small()
{
	neural::MainSignal::instance().reset_interrupt_flag();
	neural::MainSignal::instance().install_sigint_handler();
	std::cout << "CIFAR-10 small conv demo (CUDA) — Conv+BN+ReLU+Pool ×3; FC+BN+ReLU+FC\n";

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

	neural::CosineScheduling lr_schedule( 0.1f, 1e-4f, 500 );
	std::cout << "LR schedule: cosine (max=" << lr_schedule.lr_max() << " → min=" << lr_schedule.lr_min()
	          << " over " << lr_schedule.cosine_epochs() << " epochs).\n";

	// Small: 3× (Conv+BN+ReLU+MaxPool); flat = 4*4*128 = 2048;
	// FC(2048→128) + BN + ReLU; FC(128→10).
	SmallConvBox_t box(
	    3, 32, 32,
	    neural::ConvolutionalLayer<TensorN_t>( 32, 3 ),
	    neural::BatchNormLayer<TensorN_t>( 32, 1e-5f, 0.01f ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::MaxPoolLayer<TensorN_t>( 2, 2 ),
	    neural::ConvolutionalLayer<TensorN_t>( 64, 3 ),
	    neural::BatchNormLayer<TensorN_t>( 64, 1e-5f, 0.01f ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::MaxPoolLayer<TensorN_t>( 2, 2 ),
	    neural::ConvolutionalLayer<TensorN_t>( 128, 3 ),
	    neural::BatchNormLayer<TensorN_t>( 128, 1e-5f, 0.01f ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::MaxPoolLayer<TensorN_t>( 2, 2 ) );

	SmallConvNN_t nn(
	    box,
	    neural::LinearLayer<Tensor2D_t>( 128 ),
	    neural::BatchNorm1dLayer<Tensor2D_t>( 128, 1e-5f, 0.01f ),
	    neural::ReLULayer<Tensor2D_t>(),
	    neural::LinearLayer<Tensor2D_t>( 10 ) );

	neural::MomentumSGDOptimizer<Tensor2D_t, SmallConvNN_t> opt( nn );
	opt.m_learning_rate = lr_schedule.lr_max();
	opt.m_momentum      = 0.9f;
	opt.m_batch_size    = 512;
	opt.m_epochs        = 500;
	opt.m_l2_regularizer = 0.0001f;
	opt.m_nesterov      = true;

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

		    bool const stop = neural::MainSignal::instance().interrupt_requested();

		    return stop ? true : ans;
	    },
	    cifar_batch_op );
	std::cout << "\n";

	float te_acc  = 0.f;
	if ( test_data.count > 0 ) {
		set_train_mode( nn, false );
		nn.ensureBuffersForShape( opt.m_batch_size, train_data.images[0].cols() );
		te_acc = neural::compute_accuracy( nn, test_data.images, test_data.labels, 10,
		                                   opt.m_batch_size );
	}
	std::cout << std::fixed << std::setprecision( 4 ) << "Test accuracy: " << te_acc * 100.0f << "%\n";
}

#endif
