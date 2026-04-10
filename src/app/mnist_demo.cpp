#include "cross_entropy_softmax_loss.hpp"
#include "linear_layer.hpp"
#include "mnist_loader.hpp"
#include "weight_debug.hpp"
#include "relu_layer.hpp"
#include "sequential_nn.hpp"
#include "sgd_optimizer.hpp"
#include "tensor.hpp"
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <string>
#include <type_traits>
#include <chrono>

namespace {

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
using NN = neural::SequentialNN<Tensor_t, neural::SoftmaxCrossEntropyLoss<Tensor_t>,
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

float compute_accuracy( NN &nn, std::vector<Tensor_t> const &images,
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

void run_mnist_demo()
{
	std::cout << "MNIST demo with SequentialNN\n";

	std::string data_dir = "data/mnist";
	auto train_data = neural::load_mnist( data_dir, 5000 );
	if ( train_data.count == 0 ) {
		data_dir = "../data/mnist";
		train_data = neural::load_mnist( data_dir, 5000 );
	}
	if ( train_data.count == 0 ) {
		std::cout << "Could not load MNIST. Place train-images and train-labels "
		             "in data/mnist/ (gunzip first)\n";
		return;
	}

	auto test_data = neural::load_mnist_test( data_dir );
	if ( test_data.count == 0 ) {
		std::cout << "Could not load test set. Place t10k-images and t10k-labels "
		             "in data/mnist/\n";
	}

	std::cout << "Loaded " << train_data.count << " training samples";
	if ( test_data.count > 0 )
		std::cout << ", " << test_data.count << " test samples";
	std::cout << ".\n";

	auto linear1 = neural::LinearLayer<Tensor_t>( 128 );
	auto relu1 = neural::ReLULayer<Tensor_t>();
	auto linear2 = neural::LinearLayer<Tensor_t>( 64 );
	auto relu2 = neural::ReLULayer<Tensor_t>();
	auto linear3 = neural::LinearLayer<Tensor_t>( 10 );

	NN nn( linear1, relu1, linear2, relu2, linear3 );

	neural::SGDOptimizer<Tensor_t, NN> opt( nn );
	opt.m_learning_rate = 0.01f;
	opt.m_batch_size = 128;
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
		           std::string const prefix = "Epoch " + std::to_string( epoch + 1 ) + "/" +
		                                       std::to_string( epoch_max ) + " ";
		           draw_progress_bar( epoch + 1, epoch_max, loss, epoch_sec, prefix.c_str() );
		           return false;
	           } );
	std::cout << "\n";

	int linear_idx = 0;
	nn.forEachLayer( [&]( auto const &layer ) {
		using L = std::decay_t<decltype( layer )>;
		if constexpr ( std::is_same_v<L, neural::LinearLayer<Tensor_t>> ) {
			std::string const prefix = "linear[" + std::to_string( linear_idx ) + "]";
			neural::print_weight_stats( prefix + ".weights", layer.getWeights() );
			neural::print_weight_stats( prefix + ".bias", layer.getBias() );
			if ( linear_idx == 0
			     && layer.getWeights().rows() == 28 * 28 ) {
				std::filesystem::path const out =
				    std::filesystem::path( "debug_weights" ) / "layer0_mnist_grid.pgm";
				if ( neural::save_mnist_first_layer_weight_grid(
				         layer.getWeights(), out, /*grid_cols=*/8,
				         /*max_neurons=*/32 ) ) {
					std::cout << "[weights] wrote " << out.string()
					          << " (32 neurons, 8 columns)\n";
				}
			}
			++linear_idx;
		}
	} );

	float train_acc = compute_accuracy( nn, train_data.images, train_data.labels );
	std::cout << "Training accuracy: " << ( train_acc * 100.0f ) << "%\n";
	if ( test_data.count > 0 ) {
		float test_acc = compute_accuracy( nn, test_data.images, test_data.labels );
		std::cout << "Test accuracy: " << ( test_acc * 100.0f ) << "%\n";
	}
}
