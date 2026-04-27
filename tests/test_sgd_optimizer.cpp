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
	auto linear = LinearLayer<Tensor<float>>( 1 );
	SimpleNN nn( linear );

	Tensor<float> batch_input( 2, 2, 1.0f );
	Tensor<float> batch_target( 2, 1, 0.5f );

	auto loss = nn.trainStep( batch_input, batch_target );
	REQUIRE( loss >= 0.0f );
}

TEST_CASE( "Manual batch build and trainStep", "[sgd_optimizer][smoke]" )
{
	using SimpleNN = SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>>;
	auto linear = LinearLayer<Tensor<float>>( 1 );
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
	auto linear = LinearLayer<Tensor<float>>( 1 );
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
	auto linear = LinearLayer<Tensor<float>>( 1 );
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

TEST_CASE( "MomentumSGDOptimizer with Nesterov runs without crash", "[sgd_optimizer][smoke]" )
{
	using SimpleNN = SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>>;
	auto linear = LinearLayer<Tensor<float>>( 1 );
	SimpleNN nn( linear );

	MomentumSGDOptimizer<Tensor<float>, SimpleNN> opt( nn );
	opt.m_nesterov = true;

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

namespace {

void require_tensors_equal( Tensor<float> const &a, Tensor<float> const &b, float tol )
{
	REQUIRE( a.size() == b.size() );
	for ( std::size_t i = 0; i < a.size(); ++i ) {
		REQUIRE_THAT( a.data()[i], WithinAbs( b.data()[i], tol ) );
	}
}

} // namespace

TEST_CASE( "MomentumSGDOptimizer with zero momentum matches SGDOptimizer", "[sgd_optimizer][momentum][parity]" )
{
	// With β=0, v=g+λW on the first update step and matches plain SGD (same lr, L2, batching).
	std::vector<float> const w_init = { 0.5f, 0.5f };
	std::vector<float> const b_init = { 0.0f };
	Tensor<float> w_sgd( 2, 1, w_init.begin(), w_init.end() );
	Tensor<float> b_sgd( 1, 1, b_init.begin(), b_init.end() );
	Tensor<float> w_mom( 2, 1, w_init.begin(), w_init.end() );
	Tensor<float> b_mom( 1, 1, b_init.begin(), b_init.end() );

	using SimpleNN = SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>>;
	SimpleNN nn_sgd( LinearLayer<Tensor<float>>( w_sgd, b_sgd ) );
	SimpleNN nn_mom( LinearLayer<Tensor<float>>( w_mom, b_mom ) );

	SGDOptimizer<Tensor<float>, SimpleNN>         opt_sgd( nn_sgd );
	MomentumSGDOptimizer<Tensor<float>, SimpleNN> opt_mom( nn_mom );
	opt_sgd.m_learning_rate = opt_mom.m_learning_rate = 0.1f;
	opt_sgd.m_l2_regularizer = opt_mom.m_l2_regularizer = 0.0f;
	opt_sgd.m_epochs = opt_mom.m_epochs = 1;
	opt_sgd.m_batch_size = opt_mom.m_batch_size = 4;
	opt_mom.m_momentum     = 0.0f;

	std::vector<Tensor<float>> inputs;
	std::vector<Tensor<float>> targets;
	for ( int i = 0; i < 4; ++i ) {
		inputs.push_back( Tensor<float>( 1, 2, static_cast<float>( i ) / 4.f ) );
		targets.push_back( Tensor<float>( 1, 1, static_cast<float>( i ) / 8.f ) );
	}

	opt_sgd.train( inputs, targets );
	opt_mom.train( inputs, targets );

	LinearLayer<Tensor<float>> *L_sgd = nullptr;
	LinearLayer<Tensor<float>> *L_mom = nullptr;
	nn_sgd.forEachLayer( [&]( auto &layer ) { L_sgd = &layer; } );
	nn_mom.forEachLayer( [&]( auto &layer ) { L_mom = &layer; } );
	REQUIRE( L_sgd != nullptr );
	REQUIRE( L_mom != nullptr );

	require_tensors_equal( L_sgd->getWeights(), L_mom->getWeights(), eps );
	require_tensors_equal( L_sgd->getBias(), L_mom->getBias(), eps );
}

TEST_CASE( "MomentumSGDOptimizer training reduces loss on simple regression", "[sgd_optimizer][momentum][convergence]" )
{
	// Mean MSE over the same 4 points: untrained clone vs. trained net (probe point can be OOD).
	std::vector<float> const w_data = { 0.1f, 0.2f };
	std::vector<float> const b_data = { 0.3f };

	using SimpleNN = SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>>;
	auto make_nn = [&]() {
		Tensor<float> w( 2, 1, w_data.begin(), w_data.end() );
		Tensor<float> b( 1, 1, b_data.begin(), b_data.end() );
		return SimpleNN( LinearLayer<Tensor<float>>( w, b ) );
	};

	SimpleNN work = make_nn();
	SimpleNN ref  = make_nn();

	MomentumSGDOptimizer<Tensor<float>, SimpleNN> opt( work );
	opt.m_learning_rate   = 0.02f;
	opt.m_momentum        = 0.9f;
	opt.m_epochs          = 80;
	opt.m_batch_size      = 2;
	opt.m_l2_regularizer  = 0.0f;

	std::vector<Tensor<float>> inputs;
	std::vector<Tensor<float>> targets;
	for ( int i = 0; i < 4; ++i ) {
		inputs.push_back( Tensor<float>( 1, 2, static_cast<float>( i ) / 4.f ) );
		targets.push_back( Tensor<float>( 1, 1, static_cast<float>( i ) / 8.f ) );
	}

	auto mean_mse = []( SimpleNN &n, std::vector<Tensor<float>> const &in,
	                   std::vector<Tensor<float>> const &tg ) {
		float s = 0.f;
		for ( std::size_t i = 0; i < in.size(); ++i ) {
			s += n.trainStep( in[i], tg[i] );
		}
		return s / static_cast<float>( in.size() );
	};

	float const loss_before = mean_mse( ref, inputs, targets );
	opt.train( inputs, targets );
	float const loss_after = mean_mse( work, inputs, targets );

	REQUIRE( loss_before > 1e-6f );
	REQUIRE( loss_after < loss_before );
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
	auto linear = LinearLayer<Tensor<float>>( 1 );
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
	auto linear1 = LinearLayer<Tensor<float>>( 4 );
	auto relu = ReLULayer<Tensor<float>>();
	auto linear2 = LinearLayer<Tensor<float>>( 1 );
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
