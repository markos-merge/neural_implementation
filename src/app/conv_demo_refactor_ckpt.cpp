#include "conv_demo_refactor_ckpt.hpp"
#include "main.hpp"
#include <iostream>

#if !NEURAL_CUDA_ENABLED || !NEURAL_CUDNN_ENABLED

void run_conv_demo_refactor_ckpt()
{
	std::cout << "conv_demo_refactor_ckpt: build with -DNEURAL_ENABLE_CUDA=ON (cuDNN required).\n";
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
#include "refactor/dropout_layer.hpp"
#include "refactor/training_scheduler.hpp"
#include "refactor/metrics.hpp"
#include "refactor/nn_io.hpp"
#include "neural_cuda_runtime.hpp"
#include <cuda_runtime.h>
#include <omp.h>
#include <cstddef>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using neural::refactor::BatchNormLayer;
using neural::refactor::StepLR;
using neural::refactor::ConvolutionalLayer;
using neural::refactor::CrossEntropyLoss;
using neural::refactor::Cuda;
using neural::refactor::DropoutLayer;
using neural::refactor::LinearLayer;
using neural::refactor::load;
using neural::refactor::MaxPoolLayer;
using neural::refactor::MomentumSGDOptimizer;
using neural::refactor::ReLULayer;
using neural::refactor::save;
using neural::refactor::SequentialNN2;
using neural::refactor::TrainingScheduler;

SequentialNN2<float, Cuda> build_network()
{
	SequentialNN2<float, Cuda> nn;

	constexpr std::size_t kPadSameK3 = 1;
	nn.addLayer( ConvolutionalLayer<float, Cuda>(
	    32, 3, std::array<std::size_t, 3>{ 3, 32, 32 }, kPadSameK3, kPadSameK3 ) );
	nn.addLayer( BatchNormLayer<float, Cuda>() );
	nn.addLayer( ReLULayer<float, Cuda>() );
	nn.addLayer( MaxPoolLayer<float, Cuda>( 2, 2 ) );

	nn.addLayer( ConvolutionalLayer<float, Cuda>( 64, 3, kPadSameK3, kPadSameK3 ) );
	nn.addLayer( BatchNormLayer<float, Cuda>() );
	nn.addLayer( ReLULayer<float, Cuda>() );
	nn.addLayer( MaxPoolLayer<float, Cuda>( 2, 2 ) );

	nn.addLayer( ConvolutionalLayer<float, Cuda>( 128, 3, kPadSameK3, kPadSameK3 ) );
	nn.addLayer( BatchNormLayer<float, Cuda>() );
	nn.addLayer( ReLULayer<float, Cuda>() );
	nn.addLayer( MaxPoolLayer<float, Cuda>( 2, 2 ) );

	nn.addLayer( LinearLayer<float, Cuda>( 128 ) );
	nn.addLayer( BatchNormLayer<float, Cuda>() );
	nn.addLayer( ReLULayer<float, Cuda>() );
	// Serialized via nn_io (keep_prob + RNG seed); eval disables dropout via set_train_mode.
	nn.addLayer( DropoutLayer<float, Cuda>( 0.5f, 77331u ) );
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

void run_conv_demo_refactor_ckpt()
{
	neural::MainSignal::instance().reset_interrupt_flag();
	neural::MainSignal::instance().install_sigint_handler();

	std::filesystem::path const ckpt_root = "nn_checkpoints";
	std::filesystem::create_directories( ckpt_root );

	std::cout << "CIFAR-10 refactor checkpoint demo (CUDA)\n";
	std::cout << "Topology: Conv+BN+ReLU+Pool × 3; FC(128)+BN+ReLU+Dropout(0.5)+FC(10)\n";

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
	if ( test_data.count == 0 ) {
		std::cout << "No test data; need CIFAR-10 test batch for accuracy sweep.\n";
		return;
	}

	constexpr std::size_t kEpochs           = 512;
	constexpr std::size_t kBatchSize        = 64;
	constexpr std::size_t kCheckpointEvery  = 1;
	std::cout << "Checkpoints: epoch_0000 (pre-train), every " << kCheckpointEvery
	          << " epochs, and final epoch → " << ckpt_root.string() << "/\n";
	std::cout << "Batch size: " << kBatchSize << " | LR: cosine (max=0.1 → min=1e-4 over "
	          << kEpochs << " epochs).\n";

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
	                                StepLR<float>,
	                                neural::Cifar10AugmentBatchOp>;
	Opt opt( 0.0f, 0.9f, CrossEntropyLoss<float, Cuda>{} );
	opt.m_l2_regularizer = 0.0001f;

	Sched sched;
	sched.m_epochs          = kEpochs;
	sched.m_batch_size      = kBatchSize;
	sched.m_lr_schedule     = { .m_lr_initial = 0.1f, .m_step_size = 50 };
	sched.m_batch_transform = std::move( augment_op );

	set_train_mode( nn, true );

	std::vector<std::pair<std::size_t, std::string>> saved_checkpoints;
	saved_checkpoints.reserve( kEpochs / kCheckpointEvery + 2 );

	auto push_checkpoint =
	    [&]( std::size_t ep1_label, std::filesystem::path const out_path ) -> void {
		    try {
			    save( nn, out_path.string() );
			    saved_checkpoints.emplace_back( ep1_label, out_path.string() );
			    std::cout << "\n[checkpoint] saved " << out_path.string() << "\n";
		    } catch ( std::exception const &ex ) {
			    std::cout << "\n[checkpoint] failed: " << ex.what() << "\n";
		    }
	    };

	{
		std::filesystem::path const pre = ckpt_root / "epoch_0000.ncx";
		std::cout << "[checkpoint] pre-training init → " << pre.string() << "\n";
		push_checkpoint( 0U, pre );
	}

	neural::TrainingEpochProgressReporter epoch_progress;

	sched.train( opt, nn, train_data.images, train_data.labels,
	    [&]( std::size_t epoch, std::size_t tot_e, std::size_t batch,
	         std::size_t tot_b, float loss, float lr ) -> bool {
		    if ( neural::MainSignal::instance().interrupt_requested() )
			    return true;
		    epoch_progress.consume_batch( loss, batch, tot_b, epoch, tot_e, lr );
		    // End of epoch: last batch index is tot_b - 1
		    if ( batch + 1U == tot_b ) {
			    std::size_t const ep1 = epoch + 1U;
			    bool const on_grid   = ( ep1 % kCheckpointEvery == 0U );
			    bool const last_ep   = ( ep1 == kEpochs );
			    if ( on_grid || last_ep ) {
				    std::ostringstream name;
				    name << "epoch_" << std::setw( 4 ) << std::setfill( '0' ) << ep1 << ".ncx";
				    push_checkpoint( ep1, ckpt_root / name.str() );
			    }
		    }
		    return false;
	    } );
	std::cout << "\n";

	neural::cifar10_apply_normalization( train_data.images, norm_mean, norm_std );
	neural::cifar10_apply_normalization( test_data.images, norm_mean, norm_std );

	std::filesystem::path const curve_path = ckpt_root / "accuracy_vs_epoch.dat";
	std::ofstream curve( curve_path );
	if ( !curve ) {
		std::cout << "Could not write " << curve_path.string() << "\n";
		return;
	}
	curve << "# epoch train_acc test_acc   (fractions 0–1)\n";
	curve << std::setprecision( 6 );

	std::cout << std::fixed << std::setprecision( 4 );
	std::cout << "Evaluating " << saved_checkpoints.size()
	          << " checkpoint(s); writing " << curve_path.string() << "\n";

	for ( auto const &[ep, path] : saved_checkpoints ) {
		SequentialNN2<float, Cuda> eval_nn;
		try {
			load( eval_nn, path );
		} catch ( std::exception const &ex ) {
			std::cout << "load failed (" << path << "): " << ex.what() << "\n";
			continue;
		}
		set_train_mode( eval_nn, false );

		float const tr =
		    neural::refactor::compute_accuracy( eval_nn, train_data.images,
		                                        train_data.labels, kBatchSize );
		float const te =
		    neural::refactor::compute_accuracy( eval_nn, test_data.images, test_data.labels,
		                                       kBatchSize );
		curve << ep << ' ' << tr << ' ' << te << '\n';
		std::cout << "epoch " << std::setw( 4 ) << ep << "  train=" << tr * 100.0f
		          << "%  test=" << te * 100.0f << "%\n";
	}

	std::cout
	    << "\n"
	       " gnuplot — train/test accuracy PNG (scale columns 2–3 to percent):\n"
	       "   gnuplot -e \"set terminal pngcairo size 900,500 enhanced; "
	       "set output 'accuracy_train_test.png'; set grid; set xlabel 'Epoch'; "
	       "set ylabel 'Accuracy (%)'; set yrange [0:100]; set key top left; "
	       "plot '"
	    << curve_path.string()
	    << "' using 1:(\\$2*100) with lines lw 2 title 'Train', '' using "
	       "1:(\\$3*100) with lines lw 2 title 'Test'\"\n";
}

#endif
