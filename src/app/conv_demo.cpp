#include "conv_demo.hpp"
#include "cifar10_loader.hpp"
#include "convolutional_box_static.hpp"
#include "convolutional_layer.hpp"
#include "cross_entropy_softmax_loss.hpp"
#include "linear_layer.hpp"
#include "max_pool_layer.hpp"
#include "relu_layer.hpp"
#include "sequential_nn_static.hpp"
#include "sgd_optimizer.hpp"
#include "tensor.hpp"
#include "tensor_n.hpp"
#include "training_progress_bar.hpp"
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

using Tensor2D_t = neural::Tensor<float>;
using TensorN_t  = neural::TensorN<4, float>;

// CIFAR-10: 3 channels × 32 × 32 = 3072 flat input.
// ConvolutionalBox layout (all inside the box):
//   Conv(3→16, k=3) : (N, 3, 32, 32) → (N, 16, 30, 30)   [out = H - 2*(k/2) = 32-2]
//   ReLU
//   MaxPool(2, 2)   : (N, 16, 30, 30) → (N, 16, 15, 15)   [(30-2)/2+1 = 15]
//   Conv(16→32, k=3): (N, 16, 15, 15) → (N, 32, 13, 13)   [15-2]
//   ReLU
//   MaxPool(2, 2)   : (N, 32, 13, 13) → (N, 32, 6, 6)     [(13-2)/2+1 = 6]
// Flattened conv tail width is inferred inside ConvolutionalBox from layer geometry.
using ConvBox_t = neural::ConvolutionalBox_static<
    TensorN_t, Tensor2D_t,
    neural::ConvolutionalLayer<TensorN_t>,
    neural::ReLULayer<TensorN_t>,
    neural::MaxPoolLayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>,
    neural::ReLULayer<TensorN_t>,
    neural::MaxPoolLayer<TensorN_t>>;

using ConvNN_t = neural::SequentialNN_static<
    Tensor2D_t,
    neural::SoftmaxCrossEntropyLoss<Tensor2D_t>,
    ConvBox_t,
    neural::LinearLayer<Tensor2D_t>,
    neural::ReLULayer<Tensor2D_t>,
    neural::LinearLayer<Tensor2D_t>>;

std::size_t argmax( Tensor2D_t const &t )
{
	std::size_t best    = 0;
	float best_val = t( 0, 0 );
	for ( std::size_t c = 1; c < t.cols(); ++c ) {
		if ( t( 0, c ) > best_val ) {
			best_val = t( 0, c );
			best     = c;
		}
	}
	return best;
}

float compute_accuracy( ConvNN_t &nn, std::vector<Tensor2D_t> const &images,
                         std::vector<Tensor2D_t> const &labels )
{
	std::size_t correct = 0;
	for ( std::size_t i = 0; i < images.size(); ++i ) {
		auto const out  = nn.forward( images[i] );
		std::size_t pred = argmax( out );
		std::size_t gt  = 0;
		for ( std::size_t c = 0; c < 10; ++c )
			if ( labels[i]( 0, c ) > 0.5f ) { gt = c; break; }
		if ( pred == gt )
			++correct;
	}
	return static_cast<float>( correct ) / static_cast<float>( images.size() );
}

} // namespace

void run_conv_demo()
{
	std::cout << "Conv demo (CIFAR-10, CPU)\n";

	std::filesystem::path const bin_dir = neural::find_cifar10_bin_dir();
	if ( bin_dir.empty() ) {
		std::cout << "Could not find CIFAR-10 binaries. Download cifar-10-binary.tar.gz from\n"
		             "  https://www.cs.toronto.edu/~kriz/cifar.html\n"
		             "and extract so that data_batch_1.bin is under data/cifar/cifar-10-batches-bin/\n";
		return;
	}

	// max_samples == 0 → load all training images (5 × 10,000 = 50,000).
	auto train_data = neural::load_cifar10_train<Tensor2D_t>( bin_dir, 0 );
	if ( train_data.count == 0 ) {
		std::cout << "Failed to load training data from " << bin_dir.string() << "\n";
		return;
	}
	// 0 → full test_batch.bin (10,000).
	auto test_data = neural::load_cifar10_test<Tensor2D_t>( bin_dir, 0 );

	std::cout << "Loaded " << train_data.count << " training samples";
	if ( test_data.count > 0 )
		std::cout << ", " << test_data.count << " test samples";
	std::cout << ".\n";

	ConvBox_t box(
	    3, 32, 32,
	    neural::ConvolutionalLayer<TensorN_t>( 16, 3 ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::MaxPoolLayer<TensorN_t>( 2, 2 ),
	    neural::ConvolutionalLayer<TensorN_t>( 32, 3 ),
	    neural::ReLULayer<TensorN_t>(),
	    neural::MaxPoolLayer<TensorN_t>( 2, 2 ) );

	ConvNN_t nn(
	    box,
	    neural::LinearLayer<Tensor2D_t>( 256 ),
	    neural::ReLULayer<Tensor2D_t>(),
	    neural::LinearLayer<Tensor2D_t>( 10 ) );

	neural::SGDOptimizer<Tensor2D_t, ConvNN_t> opt( nn );
	opt.m_learning_rate = 0.01f;
	opt.m_batch_size    = 1024;
	opt.m_epochs        = 1000;
	opt.m_l2_regularizer = 0.0001f;

	auto last_epoch_end = std::chrono::steady_clock::now();
	float loss_sum_epoch = 0.f;
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
		    neural::draw_training_epoch_progress_bar(
		        epoch + 1, epoch_max, mean_loss, elapsed, prefix.c_str(), opt.m_learning_rate );

		    // if ( loss < 0.1 ) {
			  //   ans = true;
		    // }
		    return ans;
	    } );
	std::cout << "\n";

	float const train_acc = compute_accuracy( nn, train_data.images, train_data.labels );
	std::cout << "Training accuracy: " << ( train_acc * 100.0f ) << "%\n";
	if ( test_data.count > 0 ) {
		float const test_acc = compute_accuracy( nn, test_data.images, test_data.labels );
		std::cout << "Test accuracy: " << ( test_acc * 100.0f ) << "%\n";
	}
}
