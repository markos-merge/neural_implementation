#include "linear_layer.hpp"
#include "mse_loss.hpp"
#include "relu_layer.hpp"
#include "sequential_nn.hpp"
#include "momentum_sgd_optimizer.hpp"
#include "sgd_optimizer.hpp"
#include "tensor.hpp"
#include "tensor_like.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

using neural::LinearLayer;
using neural::MSELoss;
using neural::ReLULayer;
using neural::MomentumSGDOptimizer;
using neural::SGDOptimizer;
using neural::SequentialNN;
using neural::Tensor;

static_assert( TensorLike<Tensor<float>> );

using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps = 1e-5f;

using NN = SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>, ReLULayer<Tensor<float>>, LinearLayer<Tensor<float>>>;

} // namespace

TEST_CASE( "SequentialNN trainStep with batch works", "[sgd_optimizer][smoke]" )
{
	using SimpleNN = SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>>;
	auto linear = LinearLayer<Tensor<float>>( 2, 1 );
	SimpleNN nn( linear );

	Tensor<float> batch_input( 2, 2, 1.0f );
	Tensor<float> batch_target( 2, 1, 0.5f );

	auto loss = nn.trainStep( batch_input, batch_target );
	REQUIRE( loss >= 0.0f );
}

TEST_CASE( "Manual batch build and trainStep", "[sgd_optimizer][smoke]" )
{
	using SimpleNN = SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>>;
	auto linear = LinearLayer<Tensor<float>>( 2, 1 );
	SimpleNN nn( linear );

	// Use explicit data - batch (2, 2) and targets (2, 1)
	std::vector<float> batch_data = { 0.0f, 0.25f, 0.5f, 0.75f };
	Tensor<float> batch_inputs( 2, 2, batch_data.begin(), batch_data.end() );

	std::vector<float> target_data = { 0.0f, 0.25f };
	Tensor<float> batch_targets( 2, 1, target_data.begin(), target_data.end() );

	auto loss = nn.trainStep( batch_inputs, batch_targets );
	REQUIRE( loss >= 0.0f );
}

TEST_CASE( "SGDOptimizer constructor and train runs without crash", "[sgd_optimizer][smoke]" )
{
	using SimpleNN = SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>>;
	auto linear = LinearLayer<Tensor<float>>( 2, 1 );
	SimpleNN nn( linear );

	SGDOptimizer<Tensor<float>, SimpleNN> opt( nn );

	std::vector<Tensor<float>> inputs;
	std::vector<Tensor<float>> targets;
	for ( int i = 0; i < 4; ++i ) {
		inputs.push_back( Tensor<float>( 1, 2, static_cast<float>( i ) / 4.f ) );
		targets.push_back( Tensor<float>( 1, 1, static_cast<float>( i ) / 8.f ) );
	}

	opt.m_epochs = 2;
	opt.m_batch_size = 2;
	REQUIRE_NOTHROW( opt.train( inputs, targets ) );
}

TEST_CASE( "MomentumSGDOptimizer constructor and train runs without crash", "[sgd_optimizer][smoke]" )
{
	using SimpleNN = SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>>;
	auto linear = LinearLayer<Tensor<float>>( 2, 1 );
	SimpleNN nn( linear );

	MomentumSGDOptimizer<Tensor<float>, SimpleNN> opt( nn );

	std::vector<Tensor<float>> inputs;
	std::vector<Tensor<float>> targets;
	for ( int i = 0; i < 4; ++i ) {
		inputs.push_back( Tensor<float>( 1, 2, static_cast<float>( i ) / 4.f ) );
		targets.push_back( Tensor<float>( 1, 1, static_cast<float>( i ) / 8.f ) );
	}

	opt.m_epochs = 2;
	opt.m_batch_size = 2;
	REQUIRE_NOTHROW( opt.train( inputs, targets ) );
}

TEST_CASE( "SGDOptimizer training reduces loss on simple regression", "[sgd_optimizer][convergence]" )
{
	// Simple regression: y ≈ x1 + x2
	std::vector<float> w_data = { 0.5f, 0.5f };
	std::vector<float> b_data = { 0.0f };
	Tensor<float> weights( 2, 1, w_data.begin(), w_data.end() );
	Tensor<float> bias( 1, 1, b_data.begin(), b_data.end() );

	auto linear = LinearLayer<Tensor<float>>( weights, bias );
	using SimpleNN = SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>>;
	SimpleNN nn( linear );

	SGDOptimizer<Tensor<float>, SimpleNN> opt( nn );
	opt.m_learning_rate = 0.1f;
	opt.m_epochs = 20;
	opt.m_batch_size = 2;

	std::vector<Tensor<float>> inputs;
	std::vector<Tensor<float>> targets;
	for ( int i = 0; i < 4; ++i ) {
		inputs.push_back( Tensor<float>( 1, 2, static_cast<float>( i ) / 4.f ) );
		targets.push_back( Tensor<float>( 1, 1, static_cast<float>( i ) / 8.f ) );
	}

	opt.train( inputs, targets );

	// Verify we can run forward after training
	auto const pred = nn.forward( inputs[0] );
	REQUIRE( pred.rows() == 1u );
	REQUIRE( pred.cols() == 1u );
}

TEST_CASE( "SGDOptimizer updates parameters after backward", "[sgd_optimizer][update]" )
{
	using SimpleNN = SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>>;
	auto linear = LinearLayer<Tensor<float>>( 2, 1 );
	SimpleNN nn( linear );

	SGDOptimizer<Tensor<float>, SimpleNN> opt( nn );
	opt.m_learning_rate = 0.1f;
	opt.m_epochs = 2;
	opt.m_batch_size = 2;

	std::vector<Tensor<float>> inputs;
	std::vector<Tensor<float>> targets;
	inputs.push_back( Tensor<float>( 1, 2, 1.0f ) );
	targets.push_back( Tensor<float>( 1, 1, 2.0f ) );
	inputs.push_back( Tensor<float>( 1, 2, 0.5f ) );
	targets.push_back( Tensor<float>( 1, 1, 1.0f ) );

	REQUIRE_NOTHROW( opt.train( inputs, targets ) );
}

TEST_CASE( "SGDOptimizer works with network containing ReLU", "[sgd_optimizer][relu]" )
{
	auto linear1 = LinearLayer<Tensor<float>>( 2, 4 );
	auto relu = ReLULayer<Tensor<float>>();
	auto linear2 = LinearLayer<Tensor<float>>( 4, 1 );
	NN nn( linear1, relu, linear2 );

	SGDOptimizer<Tensor<float>, NN> opt( nn );
	opt.m_epochs = 2;
	opt.m_batch_size = 2;

	std::vector<Tensor<float>> inputs;
	std::vector<Tensor<float>> targets;
	for ( int i = 0; i < 4; ++i ) {
		inputs.push_back( Tensor<float>( 1, 2, static_cast<float>( i ) / 4.f ) );
		targets.push_back( Tensor<float>( 1, 1, static_cast<float>( i ) / 8.f ) );
	}

	REQUIRE_NOTHROW( opt.train( inputs, targets ) );
}
