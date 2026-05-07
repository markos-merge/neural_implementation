#include "refactor/sigmoid_layer.hpp"
#include "refactor/latch_layer.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <vector>
#include <cstring>
#include <cstddef>
#include <cmath>
#include <utility>
#if NEURAL_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

using neural::refactor::Cpu;
using neural::refactor::LatchLayer;
using neural::refactor::SigmoidLayer;
using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps = 1e-4f;

using SizePair = std::pair<std::size_t, std::size_t>;

static const SizePair k_shapes[]     = { {1,1}, {2,5}, {7,3}, {16,64}, {32,128} };
static const SizePair k_num_shapes[] = { {1,1}, {3,4}, {8,8}, {16,32} };

inline float sigmoid( float x ) { return 1.f / ( 1.f + std::exp( -x ) ); }

// ---- CPU wiring -----------------------------------------------------------

void wireCpu(
	SigmoidLayer<float, Cpu>& layer,
	LatchLayer<float, Cpu>&   in_latch,
	LatchLayer<float, Cpu>&   out_latch,
	LatchLayer<float, Cpu>&   g_in_latch,
	LatchLayer<float, Cpu>&   g_out_latch,
	std::vector<float> const& in_data,
	std::vector<float> const& g_in_data,
	std::size_t rows, std::size_t cols )
{
	in_latch.resize( {rows, cols} );
	std::memcpy( in_latch.fwdData(), in_data.data(), in_data.size() * sizeof(float) );

	out_latch.resize( {rows, cols} );

	g_in_latch.resize( {rows, cols} );
	std::memcpy( g_in_latch.fwdData(), g_in_data.data(), g_in_data.size() * sizeof(float) );

	layer.setInputOutputLatches( &in_latch, &out_latch );
	layer.setGradLatches( &g_in_latch, &g_out_latch );
}

#if NEURAL_CUDA_ENABLED
using neural::refactor::Cuda;

// ---- CUDA wiring ----------------------------------------------------------

void wireCuda(
	SigmoidLayer<float, Cuda>& layer,
	LatchLayer<float, Cuda>&   in_latch,
	LatchLayer<float, Cuda>&   out_latch,
	LatchLayer<float, Cuda>&   g_in_latch,
	LatchLayer<float, Cuda>&   g_out_latch,
	std::vector<float> const&  in_data,
	std::vector<float> const&  g_in_data,
	std::size_t rows, std::size_t cols )
{
	in_latch.resize( {rows, cols} );
	cudaMemcpy( in_latch.fwdData(), in_data.data(),
	            in_data.size() * sizeof(float), cudaMemcpyHostToDevice );

	out_latch.resize( {rows, cols} );

	g_in_latch.resize( {rows, cols} );
	cudaMemcpy( g_in_latch.fwdData(), g_in_data.data(),
	            g_in_data.size() * sizeof(float), cudaMemcpyHostToDevice );

	layer.setInputOutputLatches( &in_latch, &out_latch );
	layer.setGradLatches( &g_in_latch, &g_out_latch );
}
#endif

} // namespace

// ===========================================================================
// CPU tests
// ===========================================================================

TEST_CASE( "SigmoidLayer2 CPU forward output shape matches input", "[sigmoid2][cpu][forward][shape]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_shapes), std::end(k_shapes) ) );

	SigmoidLayer<float, Cpu> layer;
	LatchLayer<float, Cpu>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         std::vector<float>( n, 0.f ), std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();

	REQUIRE( out_latch.shape()[0] == rows );
	REQUIRE( out_latch.shape()[1] == cols );
}

TEST_CASE( "SigmoidLayer2 CPU forward sigma(0) = 0.5", "[sigmoid2][cpu][forward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	SigmoidLayer<float, Cpu> layer;
	LatchLayer<float, Cpu>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         std::vector<float>( n, 0.f ), std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();

	float const* out = out_latch.fwdData();
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( out[i], WithinAbs( 0.5f, eps ) );
}

TEST_CASE( "SigmoidLayer2 CPU forward output in (0, 1) for extreme inputs", "[sigmoid2][cpu][forward][numerical]" )
{
	SigmoidLayer<float, Cpu> layer;
	LatchLayer<float, Cpu>   in_latch, out_latch, g_in, g_out;

	std::vector<float> in_data  = { -10.f, 0.f, 10.f };
	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         in_data, std::vector<float>( 3, 0.f ), 1, 3 );
	layer.forward();

	float const* out = out_latch.fwdData();
	REQUIRE( out[0] > 0.f );
	REQUIRE( out[0] < 0.001f );
	REQUIRE_THAT( out[1], WithinAbs( 0.5f, eps ) );
	REQUIRE( out[2] > 0.999f );
	REQUIRE( out[2] < 1.001f );
}

TEST_CASE( "SigmoidLayer2 CPU forward arbitrary values match formula", "[sigmoid2][cpu][forward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	SigmoidLayer<float, Cpu> layer;
	LatchLayer<float, Cpu>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = static_cast<float>( i ) - static_cast<float>( n ) * 0.5f;

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         in_data, std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();

	float const* out = out_latch.fwdData();
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( out[i], WithinAbs( sigmoid( in_data[i] ), eps ) );
}

TEST_CASE( "SigmoidLayer2 CPU backward gradient shape matches input", "[sigmoid2][cpu][backward][shape]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_shapes), std::end(k_shapes) ) );

	SigmoidLayer<float, Cpu> layer;
	LatchLayer<float, Cpu>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         std::vector<float>( n, 0.f ), std::vector<float>( n, 1.f ), rows, cols );
	layer.forward();
	layer.backward();

	REQUIRE( g_out.shape()[0] == rows );
	REQUIRE( g_out.shape()[1] == cols );
}

TEST_CASE( "SigmoidLayer2 CPU backward at x=0: gradient = 0.25", "[sigmoid2][cpu][backward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	SigmoidLayer<float, Cpu> layer;
	LatchLayer<float, Cpu>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         std::vector<float>( n, 0.f ), std::vector<float>( n, 1.f ), rows, cols );
	layer.forward();
	layer.backward();

	// σ(0)*(1-σ(0)) = 0.5*0.5 = 0.25
	float const* g = g_out.fwdData();
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( g[i], WithinAbs( 0.25f, eps ) );
}

TEST_CASE( "SigmoidLayer2 CPU backward arbitrary values match formula", "[sigmoid2][cpu][backward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	SigmoidLayer<float, Cpu> layer;
	LatchLayer<float, Cpu>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	std::vector<float> g_in_data( n );
	for ( std::size_t i = 0; i < n; ++i ) {
		in_data[i]   = static_cast<float>( i ) - static_cast<float>( n ) * 0.5f;
		g_in_data[i] = static_cast<float>( i ) * 0.1f;
	}

	wireCpu( layer, in_latch, out_latch, g_in, g_out, in_data, g_in_data, rows, cols );
	layer.forward();
	layer.backward();

	// d/dx = grad_in * σ(x) * (1 - σ(x))
	float const* g = g_out.fwdData();
	for ( std::size_t i = 0; i < n; ++i ) {
		float const s        = sigmoid( in_data[i] );
		float const expected = g_in_data[i] * s * ( 1.f - s );
		REQUIRE_THAT( g[i], WithinAbs( expected, eps ) );
	}
}

// ===========================================================================
// CUDA tests
// ===========================================================================

#if NEURAL_CUDA_ENABLED

TEST_CASE( "SigmoidLayer2 CUDA forward output shape matches input", "[sigmoid2][cuda][forward][shape]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_shapes), std::end(k_shapes) ) );

	SigmoidLayer<float, Cuda> layer;
	LatchLayer<float, Cuda>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	wireCuda( layer, in_latch, out_latch, g_in, g_out,
	          std::vector<float>( n, 0.f ), std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();

	REQUIRE( out_latch.shape()[0] == rows );
	REQUIRE( out_latch.shape()[1] == cols );
}

TEST_CASE( "SigmoidLayer2 CUDA forward sigma(0) = 0.5", "[sigmoid2][cuda][forward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	SigmoidLayer<float, Cuda> layer;
	LatchLayer<float, Cuda>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	wireCuda( layer, in_latch, out_latch, g_in, g_out,
	          std::vector<float>( n, 0.f ), std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();
	cudaDeviceSynchronize();

	std::vector<float> out( n );
	cudaMemcpy( out.data(), out_latch.fwdData(), n * sizeof(float), cudaMemcpyDeviceToHost );
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( out[i], WithinAbs( 0.5f, eps ) );
}

TEST_CASE( "SigmoidLayer2 CUDA forward arbitrary values match formula", "[sigmoid2][cuda][forward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	SigmoidLayer<float, Cuda> layer;
	LatchLayer<float, Cuda>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = static_cast<float>( i ) - static_cast<float>( n ) * 0.5f;

	wireCuda( layer, in_latch, out_latch, g_in, g_out,
	          in_data, std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();
	cudaDeviceSynchronize();

	std::vector<float> out( n );
	cudaMemcpy( out.data(), out_latch.fwdData(), n * sizeof(float), cudaMemcpyDeviceToHost );
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( out[i], WithinAbs( sigmoid( in_data[i] ), eps ) );
}

TEST_CASE( "SigmoidLayer2 CUDA backward at x=0: gradient = 0.25", "[sigmoid2][cuda][backward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	SigmoidLayer<float, Cuda> layer;
	LatchLayer<float, Cuda>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	wireCuda( layer, in_latch, out_latch, g_in, g_out,
	          std::vector<float>( n, 0.f ), std::vector<float>( n, 1.f ), rows, cols );
	layer.forward();
	layer.backward();
	cudaDeviceSynchronize();

	std::vector<float> g( n );
	cudaMemcpy( g.data(), g_out.fwdData(), n * sizeof(float), cudaMemcpyDeviceToHost );
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( g[i], WithinAbs( 0.25f, eps ) );
}

TEST_CASE( "SigmoidLayer2 CUDA backward arbitrary values match formula", "[sigmoid2][cuda][backward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	SigmoidLayer<float, Cuda> layer;
	LatchLayer<float, Cuda>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	std::vector<float> g_in_data( n );
	for ( std::size_t i = 0; i < n; ++i ) {
		in_data[i]   = static_cast<float>( i ) - static_cast<float>( n ) * 0.5f;
		g_in_data[i] = static_cast<float>( i ) * 0.1f;
	}

	wireCuda( layer, in_latch, out_latch, g_in, g_out, in_data, g_in_data, rows, cols );
	layer.forward();
	layer.backward();
	cudaDeviceSynchronize();

	std::vector<float> g( n );
	cudaMemcpy( g.data(), g_out.fwdData(), n * sizeof(float), cudaMemcpyDeviceToHost );
	for ( std::size_t i = 0; i < n; ++i ) {
		float const s        = sigmoid( in_data[i] );
		float const expected = g_in_data[i] * s * ( 1.f - s );
		REQUIRE_THAT( g[i], WithinAbs( expected, eps ) );
	}
}

#endif // NEURAL_CUDA_ENABLED
