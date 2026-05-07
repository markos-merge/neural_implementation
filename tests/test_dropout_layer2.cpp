#include "refactor/dropout_layer.hpp"
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
using neural::refactor::DropoutLayer;
using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps = 1e-4f;

using SizePair = std::pair<std::size_t, std::size_t>;

static const SizePair k_shapes[]     = { {1,1}, {2,5}, {7,3}, {16,64}, {32,128} };
static const SizePair k_num_shapes[] = { {1,1}, {3,4}, {8,8}, {16,32} };

// ---- CPU wiring -----------------------------------------------------------

void wireCpu(
	DropoutLayer<float, Cpu>& layer,
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
	DropoutLayer<float, Cuda>& layer,
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

TEST_CASE( "DropoutLayer2 CPU eval forward output shape matches input", "[dropout2][cpu][forward][eval][shape]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_shapes), std::end(k_shapes) ) );
	std::size_t const n = rows * cols;

	DropoutLayer<float, Cpu> layer( 0.5f, 1u );
	layer.setTraining( false );
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         std::vector<float>( n, 1.f ), std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();

	REQUIRE( out_latch.shape()[0] == rows );
	REQUIRE( out_latch.shape()[1] == cols );
}

TEST_CASE( "DropoutLayer2 CPU eval forward is identity", "[dropout2][cpu][forward][eval]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );
	std::size_t const n = rows * cols;

	DropoutLayer<float, Cpu> layer( 0.5f, 1u );
	layer.setTraining( false );
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = static_cast<float>( i ) + 1.f;

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         in_data, std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();

	float const* out = out_latch.fwdData();
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( out[i], WithinAbs( in_data[i], eps ) );
}

TEST_CASE( "DropoutLayer2 CPU eval backward is identity", "[dropout2][cpu][backward][eval]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );
	std::size_t const n = rows * cols;

	DropoutLayer<float, Cpu> layer( 0.5f, 1u );
	layer.setTraining( false );
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;

	std::vector<float> g_in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		g_in_data[i] = static_cast<float>( i ) * 0.1f + 1.f;

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         std::vector<float>( n, 1.f ), g_in_data, rows, cols );
	layer.forward();
	layer.backward();

	float const* g = g_out.fwdData();
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( g[i], WithinAbs( g_in_data[i], eps ) );
}

TEST_CASE( "DropoutLayer2 CPU train keep_prob=1 is identity forward+backward", "[dropout2][cpu][train][keepone]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );
	std::size_t const n = rows * cols;

	DropoutLayer<float, Cpu> layer( 1.0f, 1u );
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         std::vector<float>( n, 3.5f ), std::vector<float>( n, 1.f ), rows, cols );
	layer.forward();

	float const* out = out_latch.fwdData();
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( out[i], WithinAbs( 3.5f, eps ) );

	layer.backward();
	float const* g = g_out.fwdData();
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( g[i], WithinAbs( 1.f, eps ) );
}

TEST_CASE( "DropoutLayer2 CPU train deterministic with fixed seed", "[dropout2][cpu][train][rng]" )
{
	constexpr std::size_t rows = 8, cols = 16, n = rows * cols;

	std::vector<float> in_data( n, 1.f );
	std::vector<float> zero_g( n, 0.f );

	DropoutLayer<float, Cpu> layer( 0.5f, 42u );
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;
	wireCpu( layer, in_latch, out_latch, g_in, g_out, in_data, zero_g, rows, cols );
	layer.forward();

	std::vector<float> first( n );
	std::memcpy( first.data(), out_latch.fwdData(), n * sizeof(float) );

	layer.setRngSeed( 42u );
	layer.forward();

	float const* out = out_latch.fwdData();
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( out[i], WithinAbs( first[i], eps ) );
}

TEST_CASE( "DropoutLayer2 CPU train inverted dropout preserves expectation", "[dropout2][cpu][train][stats]" )
{
	constexpr std::size_t rows = 64, cols = 128, n = rows * cols;
	constexpr float keep_prob = 0.6f;
	constexpr float inv_keep  = 1.f / keep_prob;

	DropoutLayer<float, Cpu> layer( keep_prob, 12345u );
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;
	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         std::vector<float>( n, 1.f ), std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();

	float const* out = out_latch.fwdData();
	std::size_t kept = 0;
	float sum = 0.f;
	for ( std::size_t i = 0; i < n; ++i ) {
		REQUIRE( ( out[i] == 0.f || std::abs( out[i] - inv_keep ) < eps ) );
		if ( out[i] != 0.f ) ++kept;
		sum += out[i];
	}

	// 8192 samples, binomial std-dev ≈ 0.005 — 2% tolerance is comfortable
	REQUIRE( std::abs( sum / static_cast<float>(n) - 1.f ) < 0.02f );
	REQUIRE( std::abs( static_cast<float>(kept) / static_cast<float>(n) - keep_prob ) < 0.02f );
}

TEST_CASE( "DropoutLayer2 CPU train backward routes gradient through forward mask", "[dropout2][cpu][backward][train]" )
{
	constexpr std::size_t rows = 4, cols = 8, n = rows * cols;

	DropoutLayer<float, Cpu> layer( 0.5f, 99u );
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;
	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         std::vector<float>( n, 1.f ), std::vector<float>( n, 2.f ), rows, cols );
	layer.forward();
	layer.backward();

	// Input=1 → out[i] is the scaled mask value (0 or 1/keep_prob).
	// Backward with grad_in=2: g_out[i] = 2 * out[i].
	float const* out = out_latch.fwdData();
	float const* g   = g_out.fwdData();
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( g[i], WithinAbs( 2.f * out[i], eps ) );
}

// ===========================================================================
// CUDA tests
// ===========================================================================

#if NEURAL_CUDA_ENABLED

TEST_CASE( "DropoutLayer2 CUDA eval forward is identity", "[dropout2][cuda][forward][eval]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );
	std::size_t const n = rows * cols;

	DropoutLayer<float, Cuda> layer( 0.5f, 1u );
	layer.setTraining( false );
	LatchLayer<float, Cuda> in_latch, out_latch, g_in, g_out;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = static_cast<float>( i ) + 1.f;

	wireCuda( layer, in_latch, out_latch, g_in, g_out,
	          in_data, std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();
	cudaDeviceSynchronize();

	std::vector<float> out( n );
	cudaMemcpy( out.data(), out_latch.fwdData(), n * sizeof(float), cudaMemcpyDeviceToHost );
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( out[i], WithinAbs( in_data[i], eps ) );
}

TEST_CASE( "DropoutLayer2 CUDA train keep_prob=1 is identity", "[dropout2][cuda][train][keepone]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );
	std::size_t const n = rows * cols;

	DropoutLayer<float, Cuda> layer( 1.0f, 1u );
	LatchLayer<float, Cuda> in_latch, out_latch, g_in, g_out;

	wireCuda( layer, in_latch, out_latch, g_in, g_out,
	          std::vector<float>( n, 3.5f ), std::vector<float>( n, 1.f ), rows, cols );
	layer.forward();
	cudaDeviceSynchronize();

	std::vector<float> out( n );
	cudaMemcpy( out.data(), out_latch.fwdData(), n * sizeof(float), cudaMemcpyDeviceToHost );
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( out[i], WithinAbs( 3.5f, eps ) );
}

TEST_CASE( "DropoutLayer2 CUDA train inverted dropout preserves expectation", "[dropout2][cuda][train][stats]" )
{
	constexpr std::size_t rows = 64, cols = 128, n = rows * cols;
	constexpr float keep_prob = 0.6f;
	constexpr float inv_keep  = 1.f / keep_prob;

	DropoutLayer<float, Cuda> layer( keep_prob, 12345u );
	LatchLayer<float, Cuda> in_latch, out_latch, g_in, g_out;
	wireCuda( layer, in_latch, out_latch, g_in, g_out,
	          std::vector<float>( n, 1.f ), std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();
	cudaDeviceSynchronize();

	std::vector<float> out( n );
	cudaMemcpy( out.data(), out_latch.fwdData(), n * sizeof(float), cudaMemcpyDeviceToHost );

	std::size_t kept = 0;
	float sum = 0.f;
	for ( std::size_t i = 0; i < n; ++i ) {
		REQUIRE( ( out[i] == 0.f || std::abs( out[i] - inv_keep ) < eps ) );
		if ( out[i] != 0.f ) ++kept;
		sum += out[i];
	}

	REQUIRE( std::abs( sum / static_cast<float>(n) - 1.f ) < 0.02f );
	REQUIRE( std::abs( static_cast<float>(kept) / static_cast<float>(n) - keep_prob ) < 0.02f );
}

TEST_CASE( "DropoutLayer2 CUDA train backward routes gradient through forward mask", "[dropout2][cuda][backward][train]" )
{
	constexpr std::size_t rows = 4, cols = 8, n = rows * cols;

	DropoutLayer<float, Cuda> layer( 0.5f, 7u );
	LatchLayer<float, Cuda> in_latch, out_latch, g_in, g_out;
	wireCuda( layer, in_latch, out_latch, g_in, g_out,
	          std::vector<float>( n, 1.f ), std::vector<float>( n, 2.f ), rows, cols );
	layer.forward();
	layer.backward();
	cudaDeviceSynchronize();

	std::vector<float> out_h( n ), g_h( n );
	cudaMemcpy( out_h.data(), out_latch.fwdData(), n * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( g_h.data(),   g_out.fwdData(),     n * sizeof(float), cudaMemcpyDeviceToHost );

	// Input=1 → out[i] is the scaled mask; backward: g_out[i] = grad_in * out[i]
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( g_h[i], WithinAbs( 2.f * out_h[i], eps ) );
}

#endif // NEURAL_CUDA_ENABLED
