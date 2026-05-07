#include "conv_demo_refactor.hpp"
#include "main.hpp"
#include <iostream>

#if !NEURAL_CUDA_ENABLED || !NEURAL_CUDNN_ENABLED

void run_conv_demo_refactor()
{
	std::cout << "conv_demo_refactor: build with -DNEURAL_ENABLE_CUDA=ON (cuDNN required).\n";
}

#else

#include "cifar10_loader.hpp"
#include "image_processing/image_augmentation.hpp"
#include "training_epoch_progress_reporter.hpp"
#include "tensor.hpp"
#include "refactor/sequential_nn.hpp"
#include "refactor/momentum_sgd_optimizer.hpp"
#include "refactor/cross_entropy_loss.hpp"
#include "refactor/convolutional_layer.hpp"
#include "refactor/batch_norm_layer.hpp"
#include "refactor/relu_layer.hpp"
#include "refactor/max_pool_layer.hpp"
#include "refactor/linear_layer.hpp"
#include "refactor/training_scheduler.hpp"
#include "refactor/metrics.hpp"
#include "neural_cuda_runtime.hpp"
#include <cuda_runtime.h>
#include <omp.h>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

namespace {

using neural::refactor::BatchNormLayer;
using neural::refactor::CosineAnnealingLR;
using neural::refactor::ConvolutionalLayer;
using neural::refactor::CrossEntropyLoss;
using neural::refactor::Cuda;
using neural::refactor::LinearLayer;
using neural::refactor::MaxPoolLayer;
using neural::refactor::MomentumSGDOptimizer;
using neural::refactor::ReLULayer;
using neural::refactor::SequentialNN2;
using neural::refactor::TrainingScheduler;

// ---------------------------------------------------------------------------
// Topology:
//   Conv(3→32, k3) + BN + ReLU + MaxPool(2,2)    32→16
//   Conv(32→64, k3) + BN + ReLU + MaxPool(2,2)   16→8
//   Conv(64→128, k3) + BN + ReLU + MaxPool(2,2)  8→4
//   Flatten → Linear(2048→128) + BN + ReLU → Linear(128→10)
// ---------------------------------------------------------------------------

SequentialNN2<float, Cuda> build_network()
{
	SequentialNN2<float, Cuda> nn;

	// Block 1 — entry point; reshapes flat [N, 3*32*32] → [N, 3, 32, 32]
	nn.addLayer( ConvolutionalLayer<float, Cuda>( 32, 3, { 3, 32, 32 } ) );
	nn.addLayer( BatchNormLayer<float, Cuda>() );
	nn.addLayer( ReLULayer<float, Cuda>() );
	nn.addLayer( MaxPoolLayer<float, Cuda>( 2, 2 ) );

	// Block 2
	nn.addLayer( ConvolutionalLayer<float, Cuda>( 64, 3 ) );
	nn.addLayer( BatchNormLayer<float, Cuda>() );
	nn.addLayer( ReLULayer<float, Cuda>() );
	nn.addLayer( MaxPoolLayer<float, Cuda>( 2, 2 ) );

	// Block 3
	nn.addLayer( ConvolutionalLayer<float, Cuda>( 128, 3 ) );
	nn.addLayer( BatchNormLayer<float, Cuda>() );
	nn.addLayer( ReLULayer<float, Cuda>() );
	nn.addLayer( MaxPoolLayer<float, Cuda>( 2, 2 ) );

	// Head — LinearLayer computes in_features = product(shape[1:]) lazily
	nn.addLayer( LinearLayer<float, Cuda>( 128 ) );
	nn.addLayer( BatchNormLayer<float, Cuda>() );
	nn.addLayer( ReLULayer<float, Cuda>() );
	nn.addLayer( LinearLayer<float, Cuda>( 10 ) );

	nn.wire();
	return nn;
}

void set_train_mode( SequentialNN2<float, Cuda> &nn, bool training )
{
	nn.forEachLayer( [training]( auto &l ) {
		if constexpr ( requires { l.setTraining( training ); } )
			l.setTraining( training );
	} );
}

} // namespace

void run_conv_demo_refactor()
{
	neural::MainSignal::instance().reset_interrupt_flag();
	neural::MainSignal::instance().install_sigint_handler();
	std::cout << "CIFAR-10 refactor demo (CUDA) — SequentialNN2 + MomentumSGDOptimizer\n";
	std::cout << "Topology: Conv+BN+ReLU+Pool × 3; FC(128)+BN+ReLU+FC(10)\n";

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
	constexpr std::size_t kEpochs    = 10;
	constexpr std::size_t kBatchSize = 16;
	std::cout << "LR: cosine (max=0.1 → min=1e-4 over " << kEpochs << " epochs).\n";

	// Build augmentation pipeline (one per OpenMP thread)
	std::size_t const n_threads = static_cast<std::size_t>( omp_get_max_threads() );
	std::vector<std::unique_ptr<neural::ImageAugmentation>> cifar_aug;
	cifar_aug.reserve( n_threads );
	for ( std::size_t i = 0; i < n_threads; ++i )
		cifar_aug.emplace_back( neural::make_cifar10_training_augmentation() );

	auto const [norm_mean, norm_std] = neural::cifar10_compute_normalization( train_data.images );
	neural::Cifar10AugmentBatchOp augment_op(
	    train_data.images[0].cols(), train_data.labels[0].cols(), kBatchSize,
	    std::move( cifar_aug ), norm_mean, norm_std );

	SequentialNN2<float, Cuda> nn = build_network();

	using Opt   = MomentumSGDOptimizer<float, Cuda>;
	using Sched = TrainingScheduler<float, Cuda, Opt,
	                                CosineAnnealingLR<float>,
	                                neural::Cifar10AugmentBatchOp>;
	Opt opt( 0.0f, 0.9f, CrossEntropyLoss<float, Cuda>{} );
	opt.m_l2_regularizer = 0.0001f;

	Sched sched;
	sched.m_epochs          = kEpochs;
	sched.m_batch_size      = kBatchSize;
	sched.m_lr_schedule     = { 0.1f, 1e-4f };
	sched.m_batch_transform = std::move( augment_op );

	set_train_mode( nn, true );

	neural::TrainingEpochProgressReporter epoch_progress;
	std::size_t const batches_per_epoch = train_data.count / kBatchSize;

	sched.train( opt, nn, train_data.images, train_data.labels,
	    [&]( std::size_t epoch, std::size_t tot_e,
	         std::size_t batch, std::size_t tot_b,
	         float loss, float lr ) -> bool {
	        if ( neural::MainSignal::instance().interrupt_requested() )
	            return true;
	        epoch_progress.consume_batch( loss, batch, tot_b, epoch, tot_e, lr );
	        return false;
	    } );
	std::cout << "\n";

	auto test_data = neural::load_cifar10_test<neural::Tensor<float>>( bin_dir, 0 );

	std::cout << "Loaded " << train_data.count << " training samples";
	if ( test_data.count > 0 )
		std::cout << ", " << test_data.count << " test samples";
	std::cout << ".\n";

	std::cout << "Train stats: mean=" << norm_mean << "  std=" << norm_std << "\n";
	if ( test_data.count > 0 )
		neural::cifar10_apply_normalization( test_data.images, norm_mean, norm_std );

	if ( test_data.count > 0 ) {
		set_train_mode( nn, false );
		float const te_acc = neural::refactor::compute_accuracy( nn, test_data.images, test_data.labels, kBatchSize );
		std::cout << std::fixed << std::setprecision( 4 )
		          << "Test accuracy: " << te_acc * 100.0f << "%\n";
	}
}

#endif
