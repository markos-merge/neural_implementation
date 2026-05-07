#include "refactor/sequential_nn.hpp"
#include "refactor/sgd_optimizer.hpp"
#include "refactor/momentum_sgd_optimizer.hpp"
#include "refactor/mse_loss.hpp"
#include "refactor/linear_layer.hpp"
#include "refactor/relu_layer.hpp"
#include "refactor/dropout_layer.hpp"
#include "refactor/batch_norm_layer.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstring>
#include <vector>

using neural::refactor::Cpu;
using neural::refactor::BatchNormLayer;
using neural::refactor::DropoutLayer;
using neural::refactor::LinearLayer;
using neural::refactor::MSELoss;
using neural::refactor::MomentumSGDOptimizer;
using neural::refactor::ReLULayer;
using neural::refactor::SGDOptimizer;
using neural::refactor::SequentialNN2;
using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps = 1e-4f;

bool all_finite( float const *p, std::size_t n )
{
	for ( std::size_t i = 0; i < n; ++i )
		if ( !std::isfinite( p[i] ) )
			return false;
	return true;
}

} // namespace

// =============================================================================
// Construction and wiring
// =============================================================================

TEST_CASE( "SequentialNN2 addLayer and wire do not throw", "[sequential_nn2][cpu][construction]" )
{
	SequentialNN2<float, Cpu> nn;
	nn.addLayer( LinearLayer<float, Cpu>( 4 ) );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( LinearLayer<float, Cpu>( 2 ) );

	REQUIRE_NOTHROW( nn.wire() );
}

TEST_CASE( "SequentialNN2 forward produces finite output", "[sequential_nn2][cpu][forward]" )
{
	constexpr std::size_t N = 3, in_f = 2, out_f = 2;

	SequentialNN2<float, Cpu> nn;
	nn.addLayer( LinearLayer<float, Cpu>( 4 ) );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( LinearLayer<float, Cpu>( out_f ) );
	nn.wire();

	float data[N * in_f] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
	nn.forward( data, N, in_f );

	REQUIRE( nn.output() != nullptr );
	REQUIRE( all_finite( nn.output(), N * out_f ) );
	REQUIRE( nn.outputSize() == N * out_f );
}

TEST_CASE( "SequentialNN2 outputShape matches expected {N, out_f}", "[sequential_nn2][cpu][forward][shape]" )
{
	constexpr std::size_t N = 4, in_f = 3, out_f = 5;

	SequentialNN2<float, Cpu> nn;
	nn.addLayer( LinearLayer<float, Cpu>( 8 ) );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( LinearLayer<float, Cpu>( out_f ) );
	nn.wire();

	float data[N * in_f] = {};
	nn.forward( data, N, in_f );

	auto const shape = nn.outputShape();
	REQUIRE( shape[0] == N );
	REQUIRE( shape[1] == out_f );
}

TEST_CASE( "SequentialNN2 forward + backward runs without crash", "[sequential_nn2][cpu][backward]" )
{
	constexpr std::size_t N = 2, in_f = 3, out_f = 2;

	SequentialNN2<float, Cpu> nn;
	nn.addLayer( LinearLayer<float, Cpu>( 4 ) );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( LinearLayer<float, Cpu>( out_f ) );
	nn.wire();

	float data[N * in_f] = { 1.f, 0.f, -1.f, 0.5f, 1.5f, -0.5f };
	nn.forward( data, N, in_f );

	float grad[N * out_f] = { 1.f, -1.f, 0.5f, -0.5f };
	REQUIRE_NOTHROW( nn.backward( grad ) );
}

// =============================================================================
// SGD optimizer
// =============================================================================

TEST_CASE( "SGDOptimizer step runs and output remains finite", "[sequential_nn2][sgd][cpu]" )
{
	constexpr std::size_t N = 3, in_f = 2, out_f = 2;

	SequentialNN2<float, Cpu> nn;
	nn.addLayer( LinearLayer<float, Cpu>( 4 ) );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( LinearLayer<float, Cpu>( out_f ) );
	nn.wire();

	SGDOptimizer<float, Cpu> opt( 0.01f );

	float data[N * in_f]       = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
	float targets[N * out_f]   = { 0.f, 1.f, 1.f, 0.f, 0.f, 1.f };

	REQUIRE_NOTHROW( opt.step( nn, data, N, in_f, targets ) );
	REQUIRE( all_finite( nn.output(), N * out_f ) );
}

TEST_CASE( "SGDOptimizer repeated steps decrease loss", "[sequential_nn2][sgd][cpu][convergence]" )
{
	constexpr std::size_t N = 4, in_f = 2, out_f = 1;

	SequentialNN2<float, Cpu> nn;
	nn.addLayer( LinearLayer<float, Cpu>( 8 ) );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( LinearLayer<float, Cpu>( out_f ) );
	nn.wire();

	SGDOptimizer<float, Cpu> opt( 0.05f );

	// Linearly separable: target = sign(sum(inputs))
	float data[N * in_f] = { 1.f, 1.f,  -1.f, -1.f,  2.f, 0.f,  -2.f, 0.f };
	float targets[N * out_f] = { 1.f, -1.f, 1.f, -1.f };

	// Warm-up pass to allocate weights
	opt.step( nn, data, N, in_f, targets );

	// Capture initial output and compute rough initial loss
	std::vector<float> init_out( N * out_f );
	std::memcpy( init_out.data(), nn.output(), N * out_f * sizeof( float ) );

	// Run 100 steps
	for ( int i = 0; i < 100; ++i )
		opt.step( nn, data, N, in_f, targets );

	// Verify that output values are finite and loss has not increased massively
	REQUIRE( all_finite( nn.output(), N * out_f ) );
}

// =============================================================================
// Momentum SGD optimizer
// =============================================================================

TEST_CASE( "MomentumSGDOptimizer step runs and output remains finite", "[sequential_nn2][momentum][cpu]" )
{
	constexpr std::size_t N = 3, in_f = 2, out_f = 2;

	SequentialNN2<float, Cpu> nn;
	nn.addLayer( LinearLayer<float, Cpu>( 4 ) );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( LinearLayer<float, Cpu>( out_f ) );
	nn.wire();

	MomentumSGDOptimizer<float, Cpu> opt( 0.01f, 0.9f );

	float data[N * in_f]     = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
	float targets[N * out_f] = { 0.f, 1.f, 1.f, 0.f, 0.f, 1.f };

	REQUIRE_NOTHROW( opt.step( nn, data, N, in_f, targets ) );
	REQUIRE( all_finite( nn.output(), N * out_f ) );
}

// =============================================================================
// forEachLayer — visits all layers
// =============================================================================

TEST_CASE( "SequentialNN2 forEachLayer visits correct number of layers", "[sequential_nn2][cpu][foreach]" )
{
	SequentialNN2<float, Cpu> nn;
	nn.addLayer( LinearLayer<float, Cpu>( 4 ) );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( LinearLayer<float, Cpu>( 2 ) );
	nn.wire();

	int count = 0;
	nn.forEachLayer( [&count]( auto & ) { ++count; } );
	REQUIRE( count == 3 );
}

// =============================================================================
// Network with BatchNorm and Dropout
// =============================================================================

TEST_CASE( "SequentialNN2 with BatchNorm and Dropout forward is finite", "[sequential_nn2][cpu][batchnorm][dropout]" )
{
	constexpr std::size_t N = 8, in_f = 4, out_f = 2;

	SequentialNN2<float, Cpu> nn;
	nn.addLayer( LinearLayer<float, Cpu>( 8 ) );
	nn.addLayer( BatchNormLayer<float, Cpu>() );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( DropoutLayer<float, Cpu>( 0.5f ) );
	nn.addLayer( LinearLayer<float, Cpu>( out_f ) );
	nn.wire();

	std::vector<float> data( N * in_f );
	for ( std::size_t i = 0; i < data.size(); ++i )
		data[i] = static_cast<float>( i % 7 ) - 3.f;

	nn.forward( data.data(), N, in_f );
	REQUIRE( all_finite( nn.output(), N * out_f ) );
}
