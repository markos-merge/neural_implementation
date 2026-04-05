#include "cifar10_loader.hpp"
#include "cross_entropy_softmax_loss.hpp"
#include "neural_cuda_runtime.hpp"
#include "linear_layer.hpp"
#include "relu_layer.hpp"
#include "sequential_nn.hpp"
#include "sgd_optimizer.hpp"
// #include "cuda_tensor.hpp"
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include "tensor.hpp"

namespace {

// using Tensor_t = neural::CudaTensor<float>;
using Tensor_t = neural::Tensor<float>;

int const PROGRESS_WIDTH = 40;

void draw_progress_bar( std::size_t current, std::size_t total, float loss, double epoch_sec,
                         char const *prefix = "" )
{
	float pct = total > 0 ? static_cast<float>( current ) / static_cast<float>( total ) : 0.f;
	int filled = static_cast<int>( pct * PROGRESS_WIDTH );
	std::cout << "\r" << prefix << "[";
	for ( int i = 0; i < PROGRESS_WIDTH; ++i )
		std::cout << ( i < filled ? '=' : ( i == filled ? '>' : '-' ) );
	std::cout << "] " << std::fixed << std::setprecision( 1 ) << ( pct * 100.f ) << "%"
	          << "  " << std::setprecision( 3 ) << epoch_sec << "s/epoch"
	          << "  loss: " << std::setprecision( 4 ) << loss << "    " << std::flush;
}

using CifarNN =
    neural::SequentialNN<Tensor_t, neural::SoftmaxCrossEntropyLoss<Tensor_t>,
                         neural::LinearLayer<Tensor_t>, neural::ReLULayer<Tensor_t>,
                         neural::LinearLayer<Tensor_t>, neural::ReLULayer<Tensor_t>,
                         neural::LinearLayer<Tensor_t>>;

std::size_t argmax( Tensor_t const &t )
{
	std::size_t best = 0;
	float best_val = t( 0, 0 );
	for ( std::size_t c = 1; c < t.cols(); ++c ) {
		float v = t( 0, c );
		if ( v > best_val ) {
			best_val = v;
			best = c;
		}
	}
	return best;
}

float compute_accuracy( CifarNN &nn, std::vector<Tensor_t> const &images,
                        std::vector<Tensor_t> const &labels )
{
	std::size_t correct = 0;
	for ( std::size_t i = 0; i < images.size(); ++i ) {
		auto out = nn.forward( images[i] );
		std::size_t pred = argmax( out );
		std::size_t gt = 0;
		for ( std::size_t c = 0; c < 10; ++c )
			if ( labels[i]( 0, c ) > 0.5f ) {
				gt = c;
				break;
			}
		if ( pred == gt )
			++correct;
	}
	return static_cast<float>( correct ) / static_cast<float>( images.size() );
}

} // namespace

void run_cifar10_demo()
{
	std::cout << "CIFAR-10 demo (MLP on flattened 32×32×3 images, CUDA tensors)\n";

	if ( !neural::cuda_runtime_ready() ) {
		std::cout << "CUDA runtime is not usable (no device or allocation failed); this demo needs a GPU.\n";
		return;
	}

	std::filesystem::path const bin_dir = neural::find_cifar10_bin_dir();
	if ( bin_dir.empty() ) {
		std::cout
		    << "Could not find CIFAR-10 binaries. Download cifar-10-binary.tar.gz from\n"
		       "  https://www.cs.toronto.edu/~kriz/cifar.html\n"
		       "Extract so you have e.g. data/cifar/cifar-10-batches-bin/data_batch_1.bin\n"
		       "and test_batch.bin next to each other.\n";
		return;
	}

	std::size_t const max_train = 0; // 0 = all 50k
	auto train_data = neural::load_cifar10_train<Tensor_t>( bin_dir, max_train );
	if ( train_data.count == 0 ) {
		std::cout << "Failed to read training batches from " << bin_dir.string() << "\n";
		return;
	}

	auto test_data = neural::load_cifar10_test<Tensor_t>( bin_dir, 0 );

	std::cout << "Loaded " << train_data.count << " training samples";
	if ( test_data.count > 0 ) {
		std::cout << ", " << test_data.count << " test samples";
	} else {
		std::cout << "No test samples found";
	}
	std::cout << " (input dim " << neural::cifar10_input_dim << ").\n";

	auto linear1 = neural::LinearLayer<Tensor_t>( neural::cifar10_input_dim, 256 );
	auto relu1 = neural::ReLULayer<Tensor_t>();
	auto linear2 = neural::LinearLayer<Tensor_t>( 256, 128 );
	auto relu2 = neural::ReLULayer<Tensor_t>();
	auto linear3 = neural::LinearLayer<Tensor_t>( 128, 10 );

	CifarNN nn( linear1, relu1, linear2, relu2, linear3 );

	neural::SGDOptimizer<Tensor_t, CifarNN> opt( nn );
	opt.m_learning_rate = static_cast<typename Tensor_t::value_type>(0.01);
	opt.m_batch_size = 500;
	opt.m_epochs = 5000;

	auto last_epoch_end = std::chrono::steady_clock::now();
	opt.train( train_data.images, train_data.labels,
	           [&]( std::size_t epoch, std::size_t epoch_max, std::size_t batch_idx,
	                std::size_t batch_max, float loss ) -> bool {
		           if ( batch_idx + 1 < batch_max ) {
			           return false;
		           }
		           auto const now = std::chrono::steady_clock::now();
		           double const epoch_sec =
		               std::chrono::duration<double>( now - last_epoch_end ).count();
		           last_epoch_end = now;
		           std::string const prefix = "Epoch " + std::to_string( epoch + 1 ) + "/"
		                                      + std::to_string( epoch_max ) + " ";
		           draw_progress_bar( epoch + 1, epoch_max, loss, epoch_sec, prefix.c_str() );
		           return false;
	           } );
	std::cout << "\n";

	float const train_acc = compute_accuracy( nn, train_data.images, train_data.labels );
	std::cout << "Training accuracy: " << ( train_acc * 100.0f ) << "%\n";
	if ( test_data.count > 0 ) {
		float const test_acc = compute_accuracy( nn, test_data.images, test_data.labels );
		std::cout << "Test accuracy: " << ( test_acc * 100.0f ) << "%\n";
	}
}
