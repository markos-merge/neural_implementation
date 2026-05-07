#include "refactor/sequential_nn.hpp"
#include "refactor/sgd_optimizer.hpp"
#include "refactor/momentum_sgd_optimizer.hpp"
#include "refactor/mse_loss.hpp"
#include "refactor/cross_entropy_loss.hpp"
#include "refactor/linear_layer.hpp"
#include "refactor/relu_layer.hpp"
#include "refactor/dropout_layer.hpp"
#include "refactor/batch_norm_layer.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstring>
#include <vector>

using neural::refactor::BatchNormLayer;
using neural::refactor::Cpu;
using neural::refactor::CrossEntropyLoss;
using neural::refactor::DropoutLayer;
using neural::refactor::LinearLayer;
using neural::refactor::MSELoss;
using neural::refactor::MomentumSGDOptimizer;
using neural::refactor::ReLULayer;
using neural::refactor::SGDOptimizer;
using neural::refactor::SequentialNN2;
using Catch::Matchers::WithinAbs;

namespace {

bool all_finite( float const *p, std::size_t n )
{
	for ( std::size_t i = 0; i < n; ++i )
		if ( !std::isfinite( p[i] ) )
			return false;
	return true;
}

// Build a small regression network (deterministic after first forward allocates weights)
// and run N steps, return the final output as a vector.
std::vector<float> run_sgd_steps( std::size_t steps, float lr )
{
	constexpr std::size_t N = 4, in_f = 3, out_f = 2;
	SequentialNN2<float, Cpu> nn;
	nn.addLayer( LinearLayer<float, Cpu>( 8 ) );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( LinearLayer<float, Cpu>( out_f ) );
	nn.wire();

	SGDOptimizer<float, Cpu> opt( lr );

	float data[N * in_f]     = { 1.f, -1.f, 0.f, 2.f, -2.f, 1.f, 0.5f, 0.5f, -0.5f, -1.f, 2.f, -2.f };
	float targets[N * out_f] = { 1.f, 0.f, 0.f, 1.f, 1.f, 1.f, 0.f, 0.f };

	for ( std::size_t i = 0; i < steps; ++i )
		opt.step( nn, data, N, in_f, targets );

	return std::vector<float>( nn.output(), nn.output() + N * out_f );
}

} // namespace

// =============================================================================
// Two independent runs with the same network produce finite outputs
// =============================================================================

TEST_CASE( "SGD two independent runs produce finite outputs", "[parity_cpu][sgd][finite]" )
{
	auto out_a = run_sgd_steps( 50, 0.05f );
	auto out_b = run_sgd_steps( 50, 0.05f );

	for ( float v : out_a )
		REQUIRE( std::isfinite( v ) );
	for ( float v : out_b )
		REQUIRE( std::isfinite( v ) );
}

// =============================================================================
// SGD with MSE — loss trend: mean absolute output change is positive after 100 steps
// =============================================================================

TEST_CASE( "SGD 100 steps on regression: output moves toward target", "[parity_cpu][sgd][convergence]" )
{
	constexpr std::size_t N = 4, in_f = 3, out_f = 1;

	SequentialNN2<float, Cpu> nn;
	nn.addLayer( LinearLayer<float, Cpu>( 16 ) );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( LinearLayer<float, Cpu>( out_f ) );
	nn.wire();

	SGDOptimizer<float, Cpu> opt( 0.05f );

	float data[N * in_f]     = { 1.f, 1.f, 1.f, -1.f, -1.f, -1.f, 2.f, -1.f, 0.f, -2.f, 1.f, 0.f };
	float targets[N * out_f] = { 3.f, -3.f, 1.f, -1.f };

	// Warm-up to allocate weights
	opt.step( nn, data, N, in_f, targets );

	// Capture output before training
	std::vector<float> out_init( nn.output(), nn.output() + N * out_f );

	for ( int i = 0; i < 100; ++i )
		opt.step( nn, data, N, in_f, targets );

	REQUIRE( all_finite( nn.output(), N * out_f ) );

	// Compute MSE before and after
	float mse_init = 0.f, mse_final = 0.f;
	for ( std::size_t i = 0; i < N * out_f; ++i ) {
		float d_init  = out_init[i] - targets[i];
		float d_final = nn.output()[i] - targets[i];
		mse_init  += d_init * d_init;
		mse_final += d_final * d_final;
	}
	REQUIRE( mse_final < mse_init );
}

// =============================================================================
// MomentumSGD 100 steps on regression: loss decreases
// =============================================================================

TEST_CASE( "MomentumSGD 100 steps on regression: output moves toward target",
           "[parity_cpu][momentum][convergence]" )
{
	constexpr std::size_t N = 4, in_f = 3, out_f = 1;

	SequentialNN2<float, Cpu> nn;
	nn.addLayer( LinearLayer<float, Cpu>( 16 ) );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( LinearLayer<float, Cpu>( out_f ) );
	nn.wire();

	MomentumSGDOptimizer<float, Cpu> opt( 0.02f, 0.9f );

	float data[N * in_f]     = { 1.f, 1.f, 1.f, -1.f, -1.f, -1.f, 2.f, -1.f, 0.f, -2.f, 1.f, 0.f };
	float targets[N * out_f] = { 3.f, -3.f, 1.f, -1.f };

	opt.step( nn, data, N, in_f, targets );
	std::vector<float> out_init( nn.output(), nn.output() + N * out_f );

	for ( int i = 0; i < 100; ++i )
		opt.step( nn, data, N, in_f, targets );

	REQUIRE( all_finite( nn.output(), N * out_f ) );

	float mse_init = 0.f, mse_final = 0.f;
	for ( std::size_t i = 0; i < N * out_f; ++i ) {
		float d_init  = out_init[i] - targets[i];
		float d_final = nn.output()[i] - targets[i];
		mse_init  += d_init * d_init;
		mse_final += d_final * d_final;
	}
	REQUIRE( mse_final < mse_init );
}

// =============================================================================
// CrossEntropyLoss with SGD on 4-class classification: output stays finite
// =============================================================================

TEST_CASE( "SGD with CrossEntropyLoss stays finite on 4-class problem",
           "[parity_cpu][cross_entropy][finite]" )
{
	constexpr std::size_t N = 6, in_f = 4, out_f = 4;

	SequentialNN2<float, Cpu> nn;
	nn.addLayer( LinearLayer<float, Cpu>( 16 ) );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( LinearLayer<float, Cpu>( out_f ) );
	nn.wire();

	SGDOptimizer<float, Cpu> opt( 0.05f, CrossEntropyLoss<float, Cpu>{} );

	// one-hot targets, 4 classes
	float data[N * in_f] = {
	    1.f, 0.f, 0.f, 0.f,   0.f, 1.f, 0.f, 0.f,
	    0.f, 0.f, 1.f, 0.f,   0.f, 0.f, 0.f, 1.f,
	    1.f, 1.f, 0.f, 0.f,  -1.f, 0.f, 1.f, 0.f
	};
	float targets[N * out_f] = {
	    1.f, 0.f, 0.f, 0.f,   0.f, 1.f, 0.f, 0.f,
	    0.f, 0.f, 1.f, 0.f,   0.f, 0.f, 0.f, 1.f,
	    1.f, 0.f, 0.f, 0.f,   0.f, 0.f, 1.f, 0.f
	};

	for ( int i = 0; i < 50; ++i )
		REQUIRE_NOTHROW( opt.step( nn, data, N, in_f, targets ) );

	REQUIRE( all_finite( nn.output(), N * out_f ) );
}

// =============================================================================
// Network with BatchNorm and Dropout parity: output finite after training
// =============================================================================

TEST_CASE( "SGD with BatchNorm + Dropout: output stays finite", "[parity_cpu][batchnorm][dropout]" )
{
	constexpr std::size_t N = 8, in_f = 4, out_f = 2;

	SequentialNN2<float, Cpu> nn;
	nn.addLayer( LinearLayer<float, Cpu>( 16 ) );
	nn.addLayer( BatchNormLayer<float, Cpu>() );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( DropoutLayer<float, Cpu>( 0.3f ) );
	nn.addLayer( LinearLayer<float, Cpu>( out_f ) );
	nn.wire();

	SGDOptimizer<float, Cpu> opt( 0.01f );

	std::vector<float> data( N * in_f ), targets( N * out_f );
	for ( std::size_t i = 0; i < data.size(); ++i )
		data[i] = static_cast<float>( i % 5 ) - 2.f;
	for ( std::size_t i = 0; i < targets.size(); ++i )
		targets[i] = ( i % 2 == 0 ) ? 1.f : 0.f;

	for ( int i = 0; i < 20; ++i )
		REQUIRE_NOTHROW( opt.step( nn, data.data(), N, in_f, targets.data() ) );

	REQUIRE( all_finite( nn.output(), N * out_f ) );
}
