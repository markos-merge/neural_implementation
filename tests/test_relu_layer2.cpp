#include "refactor/latch_layer.hpp"
#include "refactor/relu_layer.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <vector>
#include <cstring>
#include <cstddef>
#include <utility>
#if NEURAL_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

using neural::refactor::Cpu;
using neural::refactor::LatchLayer;
using neural::refactor::ReLULayer;
using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps = 1e-5f;

using SizePair = std::pair<std::size_t, std::size_t>;

static const SizePair k_shapes[]     = { {1,1}, {2,5}, {7,3}, {16,64}, {32,128} };
static const SizePair k_num_shapes[] = { {1,1}, {3,4}, {8,8}, {16,32} };

// ---- CPU wiring -----------------------------------------------------------

void wireCpu(
	ReLULayer<float, Cpu>&    layer,
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
	ReLULayer<float, Cuda>&   layer,
	LatchLayer<float, Cuda>&  in_latch,
	LatchLayer<float, Cuda>&  out_latch,
	LatchLayer<float, Cuda>&  g_in_latch,
	LatchLayer<float, Cuda>&  g_out_latch,
	std::vector<float> const& in_data,
	std::vector<float> const& g_in_data,
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

TEST_CASE( "ReLULayer2 CPU forward output shape matches input", "[relu2][cpu][forward][shape]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_shapes), std::end(k_shapes) ) );

	ReLULayer<float, Cpu>  layer;
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         std::vector<float>( n, 1.f ), std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();

	REQUIRE( out_latch.shape()[0] == rows );
	REQUIRE( out_latch.shape()[1] == cols );
}

TEST_CASE( "ReLULayer2 CPU forward negatives are zeroed", "[relu2][cpu][forward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	ReLULayer<float, Cpu>  layer;
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = -static_cast<float>( i + 1 );

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         in_data, std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();

	float const* out = out_latch.fwdData();
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( out[i], WithinAbs( 0.f, eps ) );
}

TEST_CASE( "ReLULayer2 CPU forward positives pass through unchanged", "[relu2][cpu][forward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	ReLULayer<float, Cpu>  layer;
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = static_cast<float>( i + 1 );

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         in_data, std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();

	float const* out = out_latch.fwdData();
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( out[i], WithinAbs( in_data[i], eps ) );
}

TEST_CASE( "ReLULayer2 CPU forward alternating signs", "[relu2][cpu][forward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	ReLULayer<float, Cpu>  layer;
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = ( i % 2 == 0 ) ?  static_cast<float>( i + 1 )
		                              : -static_cast<float>( i + 1 );

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         in_data, std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();

	float const* out = out_latch.fwdData();
	for ( std::size_t i = 0; i < n; ++i ) {
		float const expected = ( in_data[i] > 0.f ) ? in_data[i] : 0.f;
		REQUIRE_THAT( out[i], WithinAbs( expected, eps ) );
	}
}

TEST_CASE( "ReLULayer2 CPU backward gradient shape matches input", "[relu2][cpu][backward][shape]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_shapes), std::end(k_shapes) ) );

	ReLULayer<float, Cpu>  layer;
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         std::vector<float>( n, 1.f ), std::vector<float>( n, 1.f ), rows, cols );
	layer.forward();
	layer.backward();

	REQUIRE( g_out.shape()[0] == rows );
	REQUIRE( g_out.shape()[1] == cols );
}

TEST_CASE( "ReLULayer2 CPU backward gradient masked by activation", "[relu2][cpu][backward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	ReLULayer<float, Cpu>  layer;
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	std::vector<float> g_in_data( n );
	for ( std::size_t i = 0; i < n; ++i ) {
		in_data[i]   = ( i % 2 == 0 ) ?  static_cast<float>( i + 1 )
		                               : -static_cast<float>( i + 1 );
		g_in_data[i] = static_cast<float>( i ) * 0.1f;
	}

	wireCpu( layer, in_latch, out_latch, g_in, g_out, in_data, g_in_data, rows, cols );
	layer.forward();
	layer.backward();

	float const* g = g_out.fwdData();
	for ( std::size_t i = 0; i < n; ++i ) {
		float const expected = ( in_data[i] > 0.f ) ? g_in_data[i] : 0.f;
		REQUIRE_THAT( g[i], WithinAbs( expected, eps ) );
	}
}

TEST_CASE( "ReLULayer2 CPU backward all-positive passes gradient unchanged", "[relu2][cpu][backward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	ReLULayer<float, Cpu>  layer;
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	std::vector<float> g_in_data( n );
	for ( std::size_t i = 0; i < n; ++i ) {
		in_data[i]   = static_cast<float>( i + 1 );
		g_in_data[i] = static_cast<float>( i ) * 0.1f;
	}

	wireCpu( layer, in_latch, out_latch, g_in, g_out, in_data, g_in_data, rows, cols );
	layer.forward();
	layer.backward();

	float const* g = g_out.fwdData();
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( g[i], WithinAbs( g_in_data[i], eps ) );
}

TEST_CASE( "ReLULayer2 CPU backward all-negative blocks gradient", "[relu2][cpu][backward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	ReLULayer<float, Cpu>  layer;
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = -static_cast<float>( i + 1 );

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         in_data, std::vector<float>( n, 1.f ), rows, cols );
	layer.forward();
	layer.backward();

	float const* g = g_out.fwdData();
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( g[i], WithinAbs( 0.f, eps ) );
}

// Phase 2 plan: numerical agreement with analytic ReLU (matches legacy Tensor layer).

TEST_CASE( "ReLULayer2 CPU forward matches analytic reference", "[relu2][cpu][parity][forward]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	ReLULayer<float, Cpu>  layer;
	LatchLayer<float, Cpu> in_latch, out_latch, g_in_latch, g_out_latch;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] =
		    ( static_cast<float>( i % 5 ) / 17.f )
		    + ( static_cast<float>( static_cast<long>( i % 17 ) ) - 8.f );

	wireCpu( layer, in_latch, out_latch, g_in_latch, g_out_latch,
	         in_data, std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();

	float const *out = out_latch.fwdData();
	for ( std::size_t i = 0; i < n; ++i ) {
		float const reference = ( in_data[i] > 0.f ) ? in_data[i] : 0.f;
		REQUIRE_THAT( out[i], WithinAbs( reference, eps ) );
	}
}

TEST_CASE( "ReLULayer2 CPU backward matches analytic reference", "[relu2][cpu][parity][backward]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	ReLULayer<float, Cpu>  layer;
	LatchLayer<float, Cpu> in_latch, out_latch, g_in_latch, g_out_latch;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] =
		    ( static_cast<float>( i % 5 ) / 13.f )
		    + ( static_cast<float>( static_cast<long>( i % 11 ) ) - 5.f );

	std::vector<float> upstream( n );
	for ( std::size_t i = 0; i < n; ++i )
		upstream[i] = static_cast<float>( i % 7 ) * 0.03f - 0.1f;

	wireCpu( layer, in_latch, out_latch, g_in_latch, g_out_latch, in_data, upstream, rows, cols );
	layer.forward();
	layer.backward();

	float const *g_out = g_out_latch.fwdData();
	for ( std::size_t i = 0; i < n; ++i ) {
		float const mask = ( in_data[i] > 0.f ) ? 1.f : 0.f;
		float const reference = upstream[i] * mask;
		REQUIRE_THAT( g_out[i], WithinAbs( reference, eps ) );
	}
}

TEST_CASE( "ReLULayer2 CPU rank-3 latch shape forward matches flattened 2D view", "[relu2][cpu][forward][nd]" )
{
	std::vector<std::size_t> const shape = { 2, 2, 3 };
	std::size_t const flat = 2 * 2 * 3;
	std::vector<float> in_data( flat );
	for ( std::size_t i = 0; i < flat; ++i )
		in_data[i] = ( i % 2 == 0 ) ? static_cast<float>( i + 1 ) : -static_cast<float>( i + 1 );

	ReLULayer<float, Cpu>  layer;
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;
	in_latch.resize( shape );
	std::memcpy( in_latch.fwdData(), in_data.data(), flat * sizeof(float) );
	out_latch.resize( shape );
	g_in.resize( shape );
	g_out.resize( shape );
	layer.setInputOutputLatches( &in_latch, &out_latch );
	layer.setGradLatches( &g_in, &g_out );
	layer.forward();

	for ( std::size_t i = 0; i < flat; ++i ) {
		float const expected = ( in_data[i] > 0.f ) ? in_data[i] : 0.f;
		REQUIRE_THAT( out_latch.fwdData()[i], WithinAbs( expected, eps ) );
	}
}

// ===========================================================================
// CUDA tests
// ===========================================================================

#if NEURAL_CUDA_ENABLED

TEST_CASE( "ReLULayer2 CUDA forward output shape matches input", "[relu2][cuda][forward][shape]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_shapes), std::end(k_shapes) ) );

	ReLULayer<float, Cuda>  layer;
	LatchLayer<float, Cuda> in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	wireCuda( layer, in_latch, out_latch, g_in, g_out,
	          std::vector<float>( n, 1.f ), std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();

	REQUIRE( out_latch.shape()[0] == rows );
	REQUIRE( out_latch.shape()[1] == cols );
}

TEST_CASE( "ReLULayer2 CUDA forward negatives are zeroed", "[relu2][cuda][forward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	ReLULayer<float, Cuda>  layer;
	LatchLayer<float, Cuda> in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = -static_cast<float>( i + 1 );

	wireCuda( layer, in_latch, out_latch, g_in, g_out,
	          in_data, std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();
	cudaDeviceSynchronize();

	std::vector<float> out( n );
	cudaMemcpy( out.data(), out_latch.fwdData(), n * sizeof(float), cudaMemcpyDeviceToHost );
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( out[i], WithinAbs( 0.f, eps ) );
}

TEST_CASE( "ReLULayer2 CUDA forward positives pass through unchanged", "[relu2][cuda][forward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	ReLULayer<float, Cuda>  layer;
	LatchLayer<float, Cuda> in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = static_cast<float>( i + 1 );

	wireCuda( layer, in_latch, out_latch, g_in, g_out,
	          in_data, std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();
	cudaDeviceSynchronize();

	std::vector<float> out( n );
	cudaMemcpy( out.data(), out_latch.fwdData(), n * sizeof(float), cudaMemcpyDeviceToHost );
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( out[i], WithinAbs( in_data[i], eps ) );
}

TEST_CASE( "ReLULayer2 CUDA forward alternating signs", "[relu2][cuda][forward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	ReLULayer<float, Cuda>  layer;
	LatchLayer<float, Cuda> in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = ( i % 2 == 0 ) ?  static_cast<float>( i + 1 )
		                              : -static_cast<float>( i + 1 );

	wireCuda( layer, in_latch, out_latch, g_in, g_out,
	          in_data, std::vector<float>( n, 0.f ), rows, cols );
	layer.forward();
	cudaDeviceSynchronize();

	std::vector<float> out( n );
	cudaMemcpy( out.data(), out_latch.fwdData(), n * sizeof(float), cudaMemcpyDeviceToHost );
	for ( std::size_t i = 0; i < n; ++i ) {
		float const expected = ( in_data[i] > 0.f ) ? in_data[i] : 0.f;
		REQUIRE_THAT( out[i], WithinAbs( expected, eps ) );
	}
}

TEST_CASE( "ReLULayer2 CUDA backward gradient shape matches input", "[relu2][cuda][backward][shape]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_shapes), std::end(k_shapes) ) );

	ReLULayer<float, Cuda>  layer;
	LatchLayer<float, Cuda> in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	wireCuda( layer, in_latch, out_latch, g_in, g_out,
	          std::vector<float>( n, 1.f ), std::vector<float>( n, 1.f ), rows, cols );
	layer.forward();
	layer.backward();

	REQUIRE( g_out.shape()[0] == rows );
	REQUIRE( g_out.shape()[1] == cols );
}

TEST_CASE( "ReLULayer2 CUDA backward gradient masked by activation", "[relu2][cuda][backward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	ReLULayer<float, Cuda>  layer;
	LatchLayer<float, Cuda> in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	std::vector<float> g_in_data( n );
	for ( std::size_t i = 0; i < n; ++i ) {
		in_data[i]   = ( i % 2 == 0 ) ?  static_cast<float>( i + 1 )
		                               : -static_cast<float>( i + 1 );
		g_in_data[i] = static_cast<float>( i ) * 0.1f;
	}

	wireCuda( layer, in_latch, out_latch, g_in, g_out, in_data, g_in_data, rows, cols );
	layer.forward();
	layer.backward();
	cudaDeviceSynchronize();

	std::vector<float> g( n );
	cudaMemcpy( g.data(), g_out.fwdData(), n * sizeof(float), cudaMemcpyDeviceToHost );
	for ( std::size_t i = 0; i < n; ++i ) {
		float const expected = ( in_data[i] > 0.f ) ? g_in_data[i] : 0.f;
		REQUIRE_THAT( g[i], WithinAbs( expected, eps ) );
	}
}

TEST_CASE( "ReLULayer2 CUDA backward all-positive passes gradient unchanged", "[relu2][cuda][backward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	ReLULayer<float, Cuda>  layer;
	LatchLayer<float, Cuda> in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	std::vector<float> g_in_data( n );
	for ( std::size_t i = 0; i < n; ++i ) {
		in_data[i]   = static_cast<float>( i + 1 );
		g_in_data[i] = static_cast<float>( i ) * 0.1f;
	}

	wireCuda( layer, in_latch, out_latch, g_in, g_out, in_data, g_in_data, rows, cols );
	layer.forward();
	layer.backward();
	cudaDeviceSynchronize();

	std::vector<float> g( n );
	cudaMemcpy( g.data(), g_out.fwdData(), n * sizeof(float), cudaMemcpyDeviceToHost );
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( g[i], WithinAbs( g_in_data[i], eps ) );
}

TEST_CASE( "ReLULayer2 CUDA backward all-negative blocks gradient", "[relu2][cuda][backward][numerical]" )
{
	auto [rows, cols] = GENERATE( Catch::Generators::from_range( std::begin(k_num_shapes), std::end(k_num_shapes) ) );

	ReLULayer<float, Cuda>  layer;
	LatchLayer<float, Cuda> in_latch, out_latch, g_in, g_out;
	std::size_t const n = rows * cols;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = -static_cast<float>( i + 1 );

	wireCuda( layer, in_latch, out_latch, g_in, g_out,
	          in_data, std::vector<float>( n, 1.f ), rows, cols );
	layer.forward();
	layer.backward();
	cudaDeviceSynchronize();

	std::vector<float> g( n );
	cudaMemcpy( g.data(), g_out.fwdData(), n * sizeof(float), cudaMemcpyDeviceToHost );
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( g[i], WithinAbs( 0.f, eps ) );
}

#endif // NEURAL_CUDA_ENABLED
