#include "conv_demo_cuda.hpp"
#include <iostream>
#include <csignal>


#if !NEURAL_CUDA_ENABLED || !NEURAL_CUDNN_ENABLED

void run_conv_demo_cuda()
{
	std::cout << "conv_demo_cuda: build with -DNEURAL_ENABLE_CUDA=ON (cuDNN required).\n";
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
#include "relu_layer.hpp"
#include "momentum_sgd_optimizer.hpp"
#include "sequential_nn.hpp"
#include "tensor.hpp"
#include <chrono>
#include <filesystem>
#include <memory>
#include <iomanip>
#include <string>

namespace {

using Tensor2D_t = neural::CudaTensor<float>;
using TensorN_t  = neural::CudaTensor4<float>;

using ConvBox_t = neural::ConvolutionalBox<
    TensorN_t, Tensor2D_t,
    neural::ConvolutionalLayer<TensorN_t>,
    neural::BatchNormLayer<TensorN_t>,
    neural::ReLULayer<TensorN_t>,
    neural::MaxPoolLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>,
    neural::BatchNormLayer<TensorN_t>,
    neural::ReLULayer<TensorN_t>,
    neural::MaxPoolLayer<TensorN_t>>;

using ConvNN_t = neural::SequentialNN<
    Tensor2D_t,
    neural::SoftmaxCrossEntropyLoss<Tensor2D_t>,
    ConvBox_t,
    neural::LinearLayer<Tensor2D_t>,
    neural::ReLULayer<Tensor2D_t>,
    neural::LinearLayer<Tensor2D_t>>;

int const kProgressWidth = 40;

void draw_progress_bar( std::size_t current, std::size_t total, float loss,
                        double epoch_sec, char const *prefix = "" )
{
	float const pct    = total > 0 ? static_cast<float>( current ) / static_cast<float>( total ) : 0.f;
	int const filled   = static_cast<int>( pct * kProgressWidth );
	std::cout << "\r" << prefix << "[";
	for ( int i = 0; i < kProgressWidth; ++i )
		std::cout << ( i < filled ? '=' : ( i == filled ? '>' : '-' ) );
	std::cout << "] " << std::fixed << std::setprecision( 1 ) << ( pct * 100.f ) << "%"
	          << "  " << std::setprecision( 3 ) << epoch_sec << "s/epoch"
	          << "  loss: " << std::setprecision( 4 ) << loss << "    " << std::flush;
}

} // namespace

namespace {
std::sig_atomic_t volatile gSignalStatus;
}

void signal_handler(int signal)
{
    gSignalStatus = signal;
}
void run_conv_demo_cuda()
{
	std::signal( SIGINT, signal_handler );
	std::cout << "Conv demo (CIFAR-10, CUDA)\n";

	if ( !neural::cuda_runtime_ready() ) {
		std::cout << "No CUDA device available.\n";
		return;
	}

	std::filesystem::path const bin_dir = neural::find_cifar10_bin_dir();
	if ( bin_dir.empty() ) {
		std::cout << "Could not find CIFAR-10 binaries. Download cifar-10-binary.tar.gz from\n"
		             "  https://www.cs.toronto.edu/~kriz/cifar.html\n"
		             "and extract so that data_batch_1.bin is under data/cifar/cifar-10-batches-bin/\n";
		return;
	}

	auto train_data = neural::load_cifar10_train<neural::Tensor<float>>( bin_dir, 0 );
	if ( train_data.count == 0 ) {
		std::cout << "Failed to load training data from " << bin_dir.string() << "\n";
		return;
	}
	auto test_data = neural::load_cifar10_test<neural::Tensor<float>>( bin_dir, 0 );

	std::cout << "Loaded " << train_data.count << " training samples (host Tensor<float>)";
	if ( test_data.count > 0 )
		std::cout << ", " << test_data.count << " test samples (host)";
	std::cout << ".\n";
	std::cout << "Training with OpenCV CIFAR-10 augmentation (host batches).\n";

	ConvBox_t box(
	    3, 32, 32,
	    neural::ConvolutionalLayer<TensorN_t>( 16, 3 ),
	    neural::BatchNormLayer<TensorN_t>( 16, 1e-5f, 0.1f ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::MaxPoolLayer<TensorN_t>( 2, 2 ),
	    neural::ConvolutionalLayer<TensorN_t>( 32, 3 ),
	    neural::BatchNormLayer<TensorN_t>( 32, 1e-5f, 0.1f ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::MaxPoolLayer<TensorN_t>( 2, 2 ) );

	ConvNN_t nn(
	    box,
	    neural::LinearLayer<Tensor2D_t>( 256 ),
	    neural::ReLULayer<Tensor2D_t>(),
	    neural::LinearLayer<Tensor2D_t>( 10 ) );

	neural::MomentumSGDOptimizer<Tensor2D_t, ConvNN_t> opt( nn );
	opt.m_learning_rate    = 0.01f;
	opt.m_momentum         = 0.9f;
	opt.m_batch_size       = 1024;
	opt.m_epochs           = 10000;
	opt.m_l2_regularizer   = 0.0001f;

	auto last_epoch_end    = std::chrono::steady_clock::now();
	float loss_sum_epoch   = 0.f;

	std::vector<std::unique_ptr<neural::ImageAugmentation>> cifar_aug;
	cifar_aug.reserve( omp_get_max_threads() );
	for ( std::size_t i = 0; i < omp_get_max_threads(); ++i ) {
		cifar_aug.emplace_back( neural::make_cifar10_training_augmentation() );
	}
	neural::Cifar10AugmentBatchOp const cifar_batch_op(
	    train_data.images[0].cols(), train_data.labels[0].cols(), opt.m_batch_size, std::move( cifar_aug ) );

	opt.train(
	    train_data.images, train_data.labels,
	    [&]( std::size_t epoch, std::size_t epoch_max, std::size_t batch_idx,
	         std::size_t batch_max, float loss ) -> bool {
		    bool ans = false;
		    loss_sum_epoch += loss;
		    if ( batch_idx + 1 < batch_max )
			    return ans;
		    float const mean_loss =
		        loss_sum_epoch / static_cast<float>( batch_max );
		    loss_sum_epoch = 0.f;
		    auto const now = std::chrono::steady_clock::now();
		    double const elapsed =
		        std::chrono::duration<double>( now - last_epoch_end ).count();
		    last_epoch_end = now;
		    std::string const prefix = "Epoch " + std::to_string( epoch + 1 ) +
		                               "/" + std::to_string( epoch_max ) + " ";
		    draw_progress_bar( epoch + 1, epoch_max, mean_loss, elapsed,
		                       prefix.c_str() );
		    return gSignalStatus != 0 ? true : ans;
	    },
	    cifar_batch_op );
	std::cout << "\n";

	float const train_acc = neural::compute_accuracy(
	    nn, train_data.images, train_data.labels, 10, opt.m_batch_size );
	std::cout << "Training accuracy: " << ( train_acc * 100.0f ) << "%\n";
	if ( test_data.count > 0 ) {
		float const test_acc = neural::compute_accuracy(
		    nn, test_data.images, test_data.labels, 10, opt.m_batch_size );
		std::cout << "Test accuracy: " << ( test_acc * 100.0f ) << "%\n";
	}
}

#endif
