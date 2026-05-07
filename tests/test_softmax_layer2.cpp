#include "refactor/softmax_layer.hpp"
#include "refactor/latch_layer.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <vector>
#include <cstring>
#include <cstddef>
#include <cmath>
#include <numeric>
#include <utility>
#if NEURAL_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

using neural::refactor::Cpu;
using neural::refactor::LatchLayer;
using neural::refactor::SoftmaxLayer;
using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps = 1e-4f;

using SizePair = std::pair<std::size_t, std::size_t>;

static const SizePair k_shapes[]     = { {1,2}, {2,5}, {4,3}, {16,10}, {32,8} };
static const SizePair k_num_shapes[] = { {1,2}, {2,4}, {4,6}, {8,10} };

// Reference softmax per row (double precision for stability)
std::vector<float> softmaxRef( std::vector<float> const& in, std::size_t rows, std::size_t cols )
{
	std::vector<float> out( rows * cols );
	for ( std::size_t r = 0; r < rows; ++r ) {
		double sum = 0.0;
		for ( std::size_t c = 0; c < cols; ++c )
			sum += std::exp( static_cast<double>( in[r * cols + c] ) );
		for ( std::size_t c = 0; c < cols; ++c )
			out[r * cols + c] = static_cast<float>(
			    std::exp( static_cast<double>( in[r * cols + c] ) ) / sum );
	}
	return out;
}

// Reference softmax backward per row:
// g_out[r][i] = p[r][i] * (g_in[r][i] - dot(g_in[r], p[r]))
std::vector<float> softmaxBackRef(
	std::vector<float> const& in,
	std::vector<float> const& g_in,
	std::size_t rows, std::size_t cols )
{
	std::vector<float> const p = softmaxRef( in, rows, cols );
	std::vector<float> g_out( rows * cols );
	for ( std::size_t r = 0; r < rows; ++r ) {
		double dot = 0.0;
		for ( std::size_t c = 0; c < cols; ++c )
			dot += static_cast<double>( g_in[r * cols + c] ) * static_cast<double>( p[r * cols + c] );
		for ( std::size_t c = 0; c < cols; ++c )
			g_out[r * cols + c] = p[r * cols + c] * ( g_in[r * cols + c] - static_cast<float>( dot ) );
	}
	return g_out;
}

// ---- CPU wiring -----------------------------------------------------------

void wireCpu(
	SoftmaxLayer<float, Cpu>& layer,
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
	SoftmaxLayer<float, Cuda>& layer,
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

TEST_CASE( "SoftmaxLayer2 CPU forward output shape matches input", "[softmax2][cpu][forward][shape]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_shapes), std::end(k_shapes) ) );

	SoftmaxLayer<float, Cpu> layer;
	LatchLayer<float, Cpu>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         std::vector<float>( n, 0.f ), std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();

	REQUIRE( out_latch.shape()[0] == rows );
	REQUIRE( out_latch.shape()[1] == cols );
}

TEST_CASE( "SoftmaxLayer2 CPU forward rows sum to 1", "[softmax2][cpu][forward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	SoftmaxLayer<float, Cpu> layer;
	LatchLayer<float, Cpu>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = static_cast<float>( i ) - static_cast<float>( n ) * 0.5f;

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         in_data, std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();

	float const* out = out_latch.fwdData();
	for ( std::size_t r = 0; r < rows; ++r ) {
		float row_sum = 0.f;
		for ( std::size_t c = 0; c < cols; ++c )
			row_sum += out[r * cols + c];
		REQUIRE_THAT( row_sum, WithinAbs( 1.f, eps ) );
	}
}

TEST_CASE( "SoftmaxLayer2 CPU forward equal logits produce uniform distribution", "[softmax2][cpu][forward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	SoftmaxLayer<float, Cpu> layer;
	LatchLayer<float, Cpu>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         std::vector<float>( n, 0.f ), std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();

	float const expected = 1.f / static_cast<float>( cols );
	float const* out = out_latch.fwdData();
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( out[i], WithinAbs( expected, eps ) );
}

TEST_CASE( "SoftmaxLayer2 CPU forward values match reference", "[softmax2][cpu][forward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	SoftmaxLayer<float, Cpu> layer;
	LatchLayer<float, Cpu>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = static_cast<float>( i ) * 0.3f - static_cast<float>( n ) * 0.15f;

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         in_data, std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();

	std::vector<float> const ref = softmaxRef( in_data, rows, cols );
	float const* out = out_latch.fwdData();
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( out[i], WithinAbs( ref[i], eps ) );
}

TEST_CASE( "SoftmaxLayer2 CPU forward dominant logit dominates output", "[softmax2][cpu][forward][numerical]" )
{
	SoftmaxLayer<float, Cpu> layer;
	LatchLayer<float, Cpu>   in_latch, out_latch, g_in, g_out;

	std::vector<float> in_data = { -5.f, 10.f, -5.f };
	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         in_data, std::vector<float>( 3, 0.f ), 1, 3 );
	layer.forward();

	float const* out = out_latch.fwdData();
	REQUIRE( out[1] > 0.99f );
	REQUIRE( out[0] < 0.01f );
	REQUIRE( out[2] < 0.01f );
}

TEST_CASE( "SoftmaxLayer2 CPU backward gradient shape matches input", "[softmax2][cpu][backward][shape]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_shapes), std::end(k_shapes) ) );

	SoftmaxLayer<float, Cpu> layer;
	LatchLayer<float, Cpu>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         std::vector<float>( n, 0.f ), std::vector<float>( n, 1.f ), rows, cols );
	layer.forward();
	layer.backward();

	REQUIRE( g_out.shape()[0] == rows );
	REQUIRE( g_out.shape()[1] == cols );
}

TEST_CASE( "SoftmaxLayer2 CPU backward two equal logits Jacobian", "[softmax2][cpu][backward][numerical]" )
{
	SoftmaxLayer<float, Cpu> layer;
	LatchLayer<float, Cpu>   in_latch, out_latch, g_in, g_out;

	std::vector<float> in_data  = { 0.f, 0.f };
	std::vector<float> g_in_data = { 1.f, 0.f };
	wireCpu( layer, in_latch, out_latch, g_in, g_out, in_data, g_in_data, 1, 2 );
	layer.forward();
	layer.backward();

	// p = [0.5, 0.5], dot = 0.5, g_out = [0.5*(1-0.5), 0.5*(0-0.5)] = [0.25, -0.25]
	float const* g = g_out.fwdData();
	REQUIRE_THAT( g[0], WithinAbs(  0.25f, eps ) );
	REQUIRE_THAT( g[1], WithinAbs( -0.25f, eps ) );
}

TEST_CASE( "SoftmaxLayer2 CPU backward uniform logits one-hot upstream", "[softmax2][cpu][backward][numerical]" )
{
	SoftmaxLayer<float, Cpu> layer;
	LatchLayer<float, Cpu>   in_latch, out_latch, g_in, g_out;

	std::vector<float> in_data   = { 0.f, 0.f, 0.f };
	std::vector<float> g_in_data = { 1.f, 0.f, 0.f };
	wireCpu( layer, in_latch, out_latch, g_in, g_out, in_data, g_in_data, 1, 3 );
	layer.forward();
	layer.backward();

	float const y   = 1.f / 3.f;
	float const dot = y;   // dot([1,0,0], [y,y,y]) = y
	float const* g = g_out.fwdData();
	REQUIRE_THAT( g[0], WithinAbs( y * ( 1.f - dot ), eps ) );
	REQUIRE_THAT( g[1], WithinAbs( y * ( 0.f - dot ), eps ) );
	REQUIRE_THAT( g[2], WithinAbs( y * ( 0.f - dot ), eps ) );
}

TEST_CASE( "SoftmaxLayer2 CPU backward arbitrary values match reference", "[softmax2][cpu][backward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	SoftmaxLayer<float, Cpu> layer;
	LatchLayer<float, Cpu>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	std::vector<float> g_in_data( n );
	for ( std::size_t i = 0; i < n; ++i ) {
		in_data[i]   = static_cast<float>( i ) * 0.3f - static_cast<float>( n ) * 0.15f;
		g_in_data[i] = static_cast<float>( i ) * 0.1f;
	}

	wireCpu( layer, in_latch, out_latch, g_in, g_out, in_data, g_in_data, rows, cols );
	layer.forward();
	layer.backward();

	std::vector<float> const ref = softmaxBackRef( in_data, g_in_data, rows, cols );
	float const* g = g_out.fwdData();
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( g[i], WithinAbs( ref[i], eps ) );
}

// ===========================================================================
// CUDA tests
// ===========================================================================

#if NEURAL_CUDA_ENABLED

TEST_CASE( "SoftmaxLayer2 CUDA forward output shape matches input", "[softmax2][cuda][forward][shape]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_shapes), std::end(k_shapes) ) );

	SoftmaxLayer<float, Cuda> layer;
	LatchLayer<float, Cuda>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	wireCuda( layer, in_latch, out_latch, g_in, g_out,
	          std::vector<float>( n, 0.f ), std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();

	REQUIRE( out_latch.shape()[0] == rows );
	REQUIRE( out_latch.shape()[1] == cols );
}

TEST_CASE( "SoftmaxLayer2 CUDA forward rows sum to 1", "[softmax2][cuda][forward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	SoftmaxLayer<float, Cuda> layer;
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
	for ( std::size_t r = 0; r < rows; ++r ) {
		float row_sum = 0.f;
		for ( std::size_t c = 0; c < cols; ++c )
			row_sum += out[r * cols + c];
		REQUIRE_THAT( row_sum, WithinAbs( 1.f, eps ) );
	}
}

TEST_CASE( "SoftmaxLayer2 CUDA forward values match reference", "[softmax2][cuda][forward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	SoftmaxLayer<float, Cuda> layer;
	LatchLayer<float, Cuda>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = static_cast<float>( i ) * 0.3f - static_cast<float>( n ) * 0.15f;

	wireCuda( layer, in_latch, out_latch, g_in, g_out,
	          in_data, std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();
	cudaDeviceSynchronize();

	std::vector<float> out( n );
	cudaMemcpy( out.data(), out_latch.fwdData(), n * sizeof(float), cudaMemcpyDeviceToHost );
	std::vector<float> const ref = softmaxRef( in_data, rows, cols );
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( out[i], WithinAbs( ref[i], eps ) );
}

TEST_CASE( "SoftmaxLayer2 CUDA backward arbitrary values match reference", "[softmax2][cuda][backward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	SoftmaxLayer<float, Cuda> layer;
	LatchLayer<float, Cuda>   in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	std::vector<float> g_in_data( n );
	for ( std::size_t i = 0; i < n; ++i ) {
		in_data[i]   = static_cast<float>( i ) * 0.3f - static_cast<float>( n ) * 0.15f;
		g_in_data[i] = static_cast<float>( i ) * 0.1f;
	}

	wireCuda( layer, in_latch, out_latch, g_in, g_out, in_data, g_in_data, rows, cols );
	layer.forward();
	layer.backward();
	cudaDeviceSynchronize();

	std::vector<float> g( n );
	cudaMemcpy( g.data(), g_out.fwdData(), n * sizeof(float), cudaMemcpyDeviceToHost );
	std::vector<float> const ref = softmaxBackRef( in_data, g_in_data, rows, cols );
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( g[i], WithinAbs( ref[i], eps ) );
}

#endif // NEURAL_CUDA_ENABLED
