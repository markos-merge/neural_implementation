#include "dropout_layer.hpp"
#include "layer.hpp"
#include "layer_wiring_helpers.hpp"
#include "tensor.hpp"
#include "tensor_like.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

#if NEURAL_CUDNN_ENABLED
#include "cuda_tensor_n.hpp"
#include "neural_cuda_runtime.hpp"
#include <array>
#include <cmath>
#endif

using neural::DropoutLayer;
using neural::Tensor;
using neural::test::wire_layer;

static_assert( TensorLike<Tensor<float>> );
static_assert( LayerLike<DropoutLayer<Tensor<float>>, Tensor<float>> );

using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps = 1e-5f;

} // namespace

TEST_CASE( "DropoutLayer rejects invalid keep_prob", "[dropout_layer][ctor]" )
{
	REQUIRE_THROWS_AS( DropoutLayer<Tensor<float>>( 0.0f, 0u ), std::invalid_argument );
	REQUIRE_THROWS_AS( DropoutLayer<Tensor<float>>( -0.1f, 0u ), std::invalid_argument );
	REQUIRE_THROWS_AS( DropoutLayer<Tensor<float>>( 1.1f, 0u ), std::invalid_argument );
}

TEST_CASE( "DropoutLayer eval forward is identity", "[dropout_layer][forward][eval]" )
{
	DropoutLayer<Tensor<float>> layer( 0.5f, 1u );
	layer.setTraining( false );

	Tensor<float> in_buf( 2, 3, 2.0f );
	Tensor<float> out_buf( 2, 3 );
	Tensor<float> g_out( 2, 3 );
	Tensor<float> g_in( 2, 3 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	for ( std::size_t r = 0; r < 2; ++r ) {
		for ( std::size_t c = 0; c < 3; ++c ) {
			REQUIRE_THAT( out_buf( r, c ), WithinAbs( 2.0f, eps ) );
		}
	}
}

TEST_CASE( "DropoutLayer eval backward passes grad_output", "[dropout_layer][backward][eval]" )
{
	DropoutLayer<Tensor<float>> layer( 0.5f, 1u );
	layer.setTraining( false );

	Tensor<float> in_buf( 1, 4, 1.0f );
	Tensor<float> out_buf( 1, 4 );
	std::vector<float> grad = { 1.f, 2.f, 3.f, 4.f };
	Tensor<float> g_out( 1, 4, grad.begin(), grad.end() );
	Tensor<float> g_in( 1, 4 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();
	layer.backward();

	for ( std::size_t c = 0; c < 4; ++c ) {
		REQUIRE_THAT( g_in( 0, c ), WithinAbs( grad[c], eps ) );
	}
}

TEST_CASE( "DropoutLayer train keep_prob 1 is identity", "[dropout_layer][forward][train]" )
{
	DropoutLayer<Tensor<float>> layer( 1.0f, 1u );

	Tensor<float> in_buf( 2, 2 );
	in_buf.assign( 3.5f );
	Tensor<float> out_buf( 2, 2 );
	Tensor<float> g_out( 2, 2, 1.0f );
	Tensor<float> g_in( 2, 2 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	for ( std::size_t r = 0; r < 2; ++r ) {
		for ( std::size_t c = 0; c < 2; ++c ) {
			REQUIRE_THAT( out_buf( r, c ), WithinAbs( 3.5f, eps ) );
		}
	}

	layer.backward();
	for ( std::size_t r = 0; r < 2; ++r ) {
		for ( std::size_t c = 0; c < 2; ++c ) {
			REQUIRE_THAT( g_in( r, c ), WithinAbs( 1.0f, eps ) );
		}
	}
}

TEST_CASE( "DropoutLayer train deterministic with fixed seed", "[dropout_layer][train][rng]" )
{
	DropoutLayer<Tensor<float>> layer( 0.5f, 42u );

	std::vector<float> data = { 1.f, 1.f, 1.f, 1.f };
	Tensor<float> in_buf( 2, 2, data.begin(), data.end() );
	Tensor<float> out_buf( 2, 2 );
	Tensor<float> g_out( 2, 2 );
	Tensor<float> g_in( 2, 2 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	Tensor<float> first = out_buf;

	layer.setRngSeed( 42u );
	in_buf.assign( 1.f );
	layer.forward();

	for ( std::size_t r = 0; r < 2; ++r ) {
		for ( std::size_t c = 0; c < 2; ++c ) {
			REQUIRE_THAT( out_buf( r, c ), WithinAbs( first( r, c ), eps ) );
		}
	}
}

TEST_CASE( "Two consecutive DropoutLayer train pass resamples each layer",
           "[dropout_layer][train][stacked][rng]" )
{
	// in -> d1 -> mid -> d2 -> out. Each forward() must draw a new mask on each
	// layer so successive full passes (forward d1, forward d2) do not repeat the
	// same activations.
	DropoutLayer<Tensor<float>> d1( 0.5f, 11u );
	DropoutLayer<Tensor<float>> d2( 0.5f, 22u );

	std::size_t const rows = 12;
	std::size_t const cols = 16;
	Tensor<float> in_buf( rows, cols, 1.0f );
	Tensor<float> mid_buf( rows, cols );
	Tensor<float> out_buf( rows, cols );
	Tensor<float> g_mid( rows, cols );
	Tensor<float> g_out( rows, cols, 1.0f );
	Tensor<float> g_in( rows, cols );

	wire_layer( d1, in_buf, mid_buf, g_mid, g_in );
	wire_layer( d2, mid_buf, out_buf, g_out, g_mid );
	d1.forward();
	d2.forward();
	Tensor<float> first_run = out_buf;

	d1.setRngSeed( 11u );
	d2.setRngSeed( 22u );
	in_buf.assign( 1.0f );
	d1.forward();
	d2.forward();
	for ( std::size_t r = 0; r < rows; ++r ) {
		for ( std::size_t c = 0; c < cols; ++c ) {
			REQUIRE_THAT( out_buf( r, c ), WithinAbs( first_run( r, c ), eps ) );
		}
	}

	in_buf.assign( 1.0f );
	d1.forward();
	d2.forward();
	std::size_t diffs = 0;
	for ( std::size_t r = 0; r < rows; ++r ) {
		for ( std::size_t c = 0; c < cols; ++c ) {
			if ( std::abs( out_buf( r, c ) - first_run( r, c ) ) > eps ) {
				++diffs;
			}
		}
	}
	// 192 elements; two fresh masks make a full match astronomically unlikely.
	REQUIRE( diffs > 0u );
}

TEST_CASE( "DropoutLayer train backward matches mask scaling", "[dropout_layer][backward][train]" )
{
	DropoutLayer<Tensor<float>> layer( 0.5f, 99u );

	Tensor<float> in_buf( 1, 3, 1.0f );
	Tensor<float> out_buf( 1, 3 );
	Tensor<float> g_out( 1, 3 );
	g_out.assign( 2.0f );
	Tensor<float> g_in( 1, 3 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	std::vector<float> expected_in_grad( 3 );
	for ( std::size_t c = 0; c < 3; ++c ) {
		float const m = out_buf( 0, c ) * 0.5f;
		expected_in_grad[c] = 2.0f * m * 2.0f;
	}

	layer.backward();
	for ( std::size_t c = 0; c < 3; ++c ) {
		REQUIRE_THAT( g_in( 0, c ), WithinAbs( expected_in_grad[c], eps ) );
	}
}

#if NEURAL_CUDNN_ENABLED

namespace {

using neural::CudaTensor4;

constexpr float cuda_eps = 1e-5f;

/// Read a CudaTensor4 element on host via the slow single-element path.
float read_at( CudaTensor4<float> const &t, std::size_t n, std::size_t c, std::size_t h, std::size_t w )
{
	return t( std::array<std::size_t, 4>{ n, c, h, w } );
}

} // namespace

TEST_CASE( "DropoutLayer<CudaTensor4> eval forward is identity",
           "[dropout_layer][cuda4][forward][eval]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	DropoutLayer<CudaTensor4<float>> layer( 0.5f, 1u );
	layer.setTraining( false );

	std::array<std::size_t, 4> const shape = { 1u, 1u, 2u, 3u };
	CudaTensor4<float> in_buf( shape, 2.0f );
	CudaTensor4<float> out_buf( shape, 0.0f );
	CudaTensor4<float> g_out( shape, 0.0f );
	CudaTensor4<float> g_in( shape, 0.0f );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	for ( std::size_t h = 0; h < 2; ++h ) {
		for ( std::size_t w = 0; w < 3; ++w ) {
			REQUIRE_THAT( read_at( out_buf, 0, 0, h, w ), WithinAbs( 2.0f, cuda_eps ) );
		}
	}
}

TEST_CASE( "DropoutLayer<CudaTensor4> train keep_prob 1 is identity",
           "[dropout_layer][cuda4][forward][train]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	DropoutLayer<CudaTensor4<float>> layer( 1.0f, 1u );

	std::array<std::size_t, 4> const shape = { 1u, 2u, 2u, 2u };
	CudaTensor4<float> in_buf( shape, 3.5f );
	CudaTensor4<float> out_buf( shape, 0.0f );
	CudaTensor4<float> g_out( shape, 1.0f );
	CudaTensor4<float> g_in( shape, 0.0f );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	for ( std::size_t c = 0; c < 2; ++c ) {
		for ( std::size_t h = 0; h < 2; ++h ) {
			for ( std::size_t w = 0; w < 2; ++w ) {
				REQUIRE_THAT( read_at( out_buf, 0, c, h, w ), WithinAbs( 3.5f, cuda_eps ) );
			}
		}
	}

	layer.backward();
	for ( std::size_t c = 0; c < 2; ++c ) {
		for ( std::size_t h = 0; h < 2; ++h ) {
			for ( std::size_t w = 0; w < 2; ++w ) {
				REQUIRE_THAT( read_at( g_in, 0, c, h, w ), WithinAbs( 1.0f, cuda_eps ) );
			}
		}
	}
}

TEST_CASE( "DropoutLayer<CudaTensor4> train preserves expectation (inverted dropout)",
           "[dropout_layer][cuda4][train][stats]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	DropoutLayer<CudaTensor4<float>> layer( 0.6f, 12345u );

	// Large enough to make the sample-mean estimate of the kept fraction tight.
	std::array<std::size_t, 4> const shape = { 4u, 8u, 16u, 16u };
	std::size_t const n =
	    shape[0] * shape[1] * shape[2] * shape[3];
	CudaTensor4<float> in_buf( shape, 1.0f );
	CudaTensor4<float> out_buf( shape, 0.0f );
	CudaTensor4<float> g_out( shape, 0.0f );
	CudaTensor4<float> g_in( shape, 0.0f );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	std::vector<float> host( n );
	if ( cudaMemcpy( host.data(), out_buf.data(), n * sizeof( float ),
	                 cudaMemcpyDeviceToHost ) != cudaSuccess ) {
		FAIL( "device->host copy failed" );
	}

	// Inverted dropout: each element is either 0 or 1/keep_prob, with E[out] = 1.
	float const inv_keep = 1.0f / 0.6f;
	std::size_t kept = 0;
	float sum = 0.0f;
	for ( float v : host ) {
		REQUIRE( ( v == 0.0f || std::abs( v - inv_keep ) < cuda_eps ) );
		if ( v != 0.0f ) {
			++kept;
		}
		sum += v;
	}

	float const mean = sum / static_cast<float>( n );
	float const kept_fraction = static_cast<float>( kept ) / static_cast<float>( n );
	// 8192 samples gives a binomial std. dev. of ~ sqrt(0.6*0.4/8192) ≈ 0.0054 for the
	// kept fraction, so tolerances of ~1.5% comfortably cover the sampling noise.
	REQUIRE( std::abs( mean - 1.0f ) < 0.02f );
	REQUIRE( std::abs( kept_fraction - 0.6f ) < 0.02f );
}

TEST_CASE( "DropoutLayer<CudaTensor4> train backward routes grad through forward mask",
           "[dropout_layer][cuda4][backward][train]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	DropoutLayer<CudaTensor4<float>> layer( 0.5f, 7u );

	std::array<std::size_t, 4> const shape = { 2u, 3u, 4u, 4u };
	std::size_t const n = shape[0] * shape[1] * shape[2] * shape[3];
	CudaTensor4<float> in_buf( shape, 1.0f );
	CudaTensor4<float> out_buf( shape, 0.0f );
	CudaTensor4<float> g_out( shape, 2.0f );
	CudaTensor4<float> g_in( shape, 0.0f );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();
	layer.backward();

	std::vector<float> host_out( n );
	std::vector<float> host_gin( n );
	if ( cudaMemcpy( host_out.data(), out_buf.data(), n * sizeof( float ),
	                 cudaMemcpyDeviceToHost ) != cudaSuccess ) {
		FAIL( "device->host copy (out) failed" );
	}
	if ( cudaMemcpy( host_gin.data(), g_in.data(), n * sizeof( float ),
	                 cudaMemcpyDeviceToHost ) != cudaSuccess ) {
		FAIL( "device->host copy (g_in) failed" );
	}

	// With input == 1, forward gives out == mask. Backward with grad_out == 2 must give
	// grad_in == 2 * mask, i.e. grad_in == 2 * out element-wise.
	for ( std::size_t i = 0; i < n; ++i ) {
		REQUIRE_THAT( host_gin[i], WithinAbs( 2.0f * host_out[i], cuda_eps ) );
	}
}

TEST_CASE( "DropoutLayer<CudaTensor4> train deterministic with fixed seed",
           "[dropout_layer][cuda4][train][rng]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::array<std::size_t, 4> const shape = { 2u, 2u, 4u, 4u };
	std::size_t const n = shape[0] * shape[1] * shape[2] * shape[3];

	auto run = [&]() {
		DropoutLayer<CudaTensor4<float>> layer( 0.5f, 4242u );
		CudaTensor4<float> in_buf( shape, 1.0f );
		CudaTensor4<float> out_buf( shape, 0.0f );
		CudaTensor4<float> g_out( shape, 0.0f );
		CudaTensor4<float> g_in( shape, 0.0f );
		wire_layer( layer, in_buf, out_buf, g_out, g_in );
		layer.forward();

		std::vector<float> host( n );
		if ( cudaMemcpy( host.data(), out_buf.data(), n * sizeof( float ),
		                 cudaMemcpyDeviceToHost ) != cudaSuccess ) {
			FAIL( "device->host copy failed" );
		}
		return host;
	};

	auto const a = run();
	auto const b = run();
	REQUIRE( a.size() == b.size() );
	for ( std::size_t i = 0; i < a.size(); ++i ) {
		REQUIRE_THAT( a[i], WithinAbs( b[i], cuda_eps ) );
	}
}

TEST_CASE( "DropoutLayer<CudaTensor4> two consecutive train pass resamples each layer",
           "[dropout_layer][cuda4][train][stacked][rng]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	// in -> d1 -> mid -> d2 -> out. Each forward() must draw a new mask on each layer
	// so successive full passes (forward d1, forward d2) do not repeat the same
	// activations.
	DropoutLayer<CudaTensor4<float>> d1( 0.5f, 11u );
	DropoutLayer<CudaTensor4<float>> d2( 0.5f, 22u );

	std::array<std::size_t, 4> const shape = { 1u, 1u, 12u, 16u };
	std::size_t const n = shape[0] * shape[1] * shape[2] * shape[3];
	std::vector<float> const ones( n, 1.0f );

	CudaTensor4<float> in_buf( shape, 1.0f );
	CudaTensor4<float> mid_buf( shape, 0.0f );
	CudaTensor4<float> out_buf( shape, 0.0f );
	CudaTensor4<float> g_mid( shape, 0.0f );
	CudaTensor4<float> g_out( shape, 1.0f );
	CudaTensor4<float> g_in( shape, 0.0f );
	wire_layer( d1, in_buf, mid_buf, g_mid, g_in );
	wire_layer( d2, mid_buf, out_buf, g_out, g_mid );

	d1.forward();
	d2.forward();
	std::vector<float> first_run( n );
	if ( cudaMemcpy( first_run.data(), out_buf.data(), n * sizeof( float ),
	                 cudaMemcpyDeviceToHost ) != cudaSuccess ) {
		FAIL( "device->host copy (first_run) failed" );
	}

	d1.setRngSeed( 11u );
	d2.setRngSeed( 22u );
	in_buf.assign( ones );
	d1.forward();
	d2.forward();
	std::vector<float> replay( n );
	if ( cudaMemcpy( replay.data(), out_buf.data(), n * sizeof( float ),
	                 cudaMemcpyDeviceToHost ) != cudaSuccess ) {
		FAIL( "device->host copy (replay) failed" );
	}
	for ( std::size_t i = 0; i < n; ++i ) {
		REQUIRE_THAT( replay[i], WithinAbs( first_run[i], cuda_eps ) );
	}

	in_buf.assign( ones );
	d1.forward();
	d2.forward();
	std::vector<float> third( n );
	if ( cudaMemcpy( third.data(), out_buf.data(), n * sizeof( float ),
	                 cudaMemcpyDeviceToHost ) != cudaSuccess ) {
		FAIL( "device->host copy (third) failed" );
	}
	std::size_t diffs = 0;
	for ( std::size_t i = 0; i < n; ++i ) {
		if ( std::abs( third[i] - first_run[i] ) > cuda_eps ) {
			++diffs;
		}
	}
	REQUIRE( diffs > 0u );
}

#endif // NEURAL_CUDNN_ENABLED
