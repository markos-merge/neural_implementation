#include "refactor/batch_norm_layer.hpp"
#include "refactor/latch_layer.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <array>
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
using neural::refactor::BatchNormLayer;
using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps_tight = 1e-5f;
constexpr float eps_loose = 1e-3f;

using SizePair = std::pair<std::size_t, std::size_t>;

// N × C pairs (batch × features) — rank-2 latch {N,C}
static const SizePair k_nc_shapes[] = { {4,2}, {8,3}, {16,4}, {32,8} };

// Rank-4 latch {N,C,H,W} (contiguous NCHW in latch buffer)
using NchwShape = std::array<std::size_t, 4>;
static const NchwShape k_nchw_shapes[] = {
    { 2, 3, 2, 2 },
    { 4, 2, 1, 5 },
    { 3, 4, 3, 1 },
};

float col_mean( float const* data, std::size_t N, std::size_t C, std::size_t c )
{
	float s = 0.f;
	for ( std::size_t n = 0; n < N; ++n )
		s += data[n * C + c];
	return s / static_cast<float>( N );
}

float col_var( float const* data, std::size_t N, std::size_t C, std::size_t c )
{
	float const mu = col_mean( data, N, C, c );
	float v = 0.f;
	for ( std::size_t n = 0; n < N; ++n ) {
		float const d = data[n * C + c] - mu;
		v += d * d;
	}
	return v / static_cast<float>( N );
}

// Contiguous NCHW: idx(n,c,h,w) = ((n * C + c) * H + h) * W + w.

float elem_nchw( float const* data, std::size_t /*N*/, std::size_t C, std::size_t H, std::size_t W,
                 std::size_t n, std::size_t c, std::size_t h, std::size_t w )
{
	return data[( ( n * C + c ) * H + h ) * W + w];
}

float ch_mean_nchw( float const* data, std::size_t N, std::size_t C, std::size_t H, std::size_t W,
                    std::size_t c )
{
	std::size_t const M = N * H * W;
	double s = 0.;
	for ( std::size_t n = 0; n < N; ++n )
		for ( std::size_t h = 0; h < H; ++h )
			for ( std::size_t w = 0; w < W; ++w )
				s += static_cast<double>( elem_nchw( data, N, C, H, W, n, c, h, w ) );
	return static_cast<float>( s / static_cast<double>( M ) );
}

float ch_var_nchw( float const* data, std::size_t N, std::size_t C, std::size_t H, std::size_t W,
                   std::size_t c )
{
	float const mu = ch_mean_nchw( data, N, C, H, W, c );
	double v = 0.;
	std::size_t const M = N * H * W;
	for ( std::size_t n = 0; n < N; ++n )
		for ( std::size_t h = 0; h < H; ++h )
			for ( std::size_t w = 0; w < W; ++w ) {
				double const d = static_cast<double>( elem_nchw( data, N, C, H, W, n, c, h, w ) ) -
				                 static_cast<double>( mu );
				v += d * d;
			}
	return static_cast<float>( v / static_cast<double>( M ) );
}

float dx_sum_ch_nchw( float const* dx, std::size_t N, std::size_t C, std::size_t H, std::size_t W,
                      std::size_t c )
{
	float s = 0.f;
	for ( std::size_t n = 0; n < N; ++n )
		for ( std::size_t h = 0; h < H; ++h )
			for ( std::size_t w = 0; w < W; ++w )
				s += elem_nchw( dx, N, C, H, W, n, c, h, w );
	return s;
}

// ---- CPU wiring -----------------------------------------------------------

void wireCpu(
	BatchNormLayer<float, Cpu>& layer,
	LatchLayer<float, Cpu>&       in_latch,
	LatchLayer<float, Cpu>&       out_latch,
	LatchLayer<float, Cpu>&       g_in_latch,
	LatchLayer<float, Cpu>&       g_out_latch,
	std::vector<float> const&     in_data,
	std::vector<float> const&     g_in_data,
	std::size_t N, std::size_t C )
{
	in_latch.resize( {N, C} );
	std::memcpy( in_latch.fwdData(), in_data.data(), in_data.size() * sizeof(float) );

	out_latch.resize( {N, C} );

	g_in_latch.resize( {N, C} );
	std::memcpy( g_in_latch.fwdData(), g_in_data.data(), g_in_data.size() * sizeof(float) );

	layer.setInputOutputLatches( &in_latch, &out_latch );
	layer.setGradLatches( &g_in_latch, &g_out_latch );
}

void wireCpuNchw(
	BatchNormLayer<float, Cpu>& layer,
	LatchLayer<float, Cpu>&     in_latch,
	LatchLayer<float, Cpu>&     out_latch,
	LatchLayer<float, Cpu>&     g_in_latch,
	LatchLayer<float, Cpu>&     g_out_latch,
	std::vector<float> const&   in_data,
	std::vector<float> const&   g_in_data,
	std::size_t N, std::size_t C, std::size_t H, std::size_t W )
{
	in_latch.resize( {N, C, H, W} );
	std::memcpy( in_latch.fwdData(), in_data.data(), in_data.size() * sizeof(float) );

	out_latch.resize( {N, C, H, W} );

	g_in_latch.resize( {N, C, H, W} );
	std::memcpy( g_in_latch.fwdData(), g_in_data.data(), g_in_data.size() * sizeof(float) );

	layer.setInputOutputLatches( &in_latch, &out_latch );
	layer.setGradLatches( &g_in_latch, &g_out_latch );
}

#if NEURAL_CUDA_ENABLED
using neural::refactor::Cuda;

// ---- CUDA wiring ----------------------------------------------------------

void wireCuda(
	BatchNormLayer<float, Cuda>& layer,
	LatchLayer<float, Cuda>&       in_latch,
	LatchLayer<float, Cuda>&       out_latch,
	LatchLayer<float, Cuda>&       g_in_latch,
	LatchLayer<float, Cuda>&       g_out_latch,
	std::vector<float> const&      in_data,
	std::vector<float> const&      g_in_data,
	std::size_t N, std::size_t C )
{
	in_latch.resize( {N, C} );
	cudaMemcpy( in_latch.fwdData(), in_data.data(),
	            in_data.size() * sizeof(float), cudaMemcpyHostToDevice );

	out_latch.resize( {N, C} );

	g_in_latch.resize( {N, C} );
	cudaMemcpy( g_in_latch.fwdData(), g_in_data.data(),
	            g_in_data.size() * sizeof(float), cudaMemcpyHostToDevice );

	layer.setInputOutputLatches( &in_latch, &out_latch );
	layer.setGradLatches( &g_in_latch, &g_out_latch );
}

void wireCudaNchw(
	BatchNormLayer<float, Cuda>& layer,
	LatchLayer<float, Cuda>&     in_latch,
	LatchLayer<float, Cuda>&     out_latch,
	LatchLayer<float, Cuda>&     g_in_latch,
	LatchLayer<float, Cuda>&     g_out_latch,
	std::vector<float> const&    in_data,
	std::vector<float> const&    g_in_data,
	std::size_t N, std::size_t C, std::size_t H, std::size_t W )
{
	in_latch.resize( {N, C, H, W} );
	cudaMemcpy( in_latch.fwdData(), in_data.data(),
	            in_data.size() * sizeof(float), cudaMemcpyHostToDevice );

	out_latch.resize( {N, C, H, W} );

	g_in_latch.resize( {N, C, H, W} );
	cudaMemcpy( g_in_latch.fwdData(), g_in_data.data(),
	            g_in_data.size() * sizeof(float), cudaMemcpyHostToDevice );

	layer.setInputOutputLatches( &in_latch, &out_latch );
	layer.setGradLatches( &g_in_latch, &g_out_latch );
}
#endif

} // namespace

// ===========================================================================
// CPU tests — rank-2 latch {N,C}
// ===========================================================================

TEST_CASE( "BatchNormLayer2 rank2 CPU forward output shape matches input",
           "[bn1d2][rank2][cpu][forward][shape]" )
{
	auto [N, C] = GENERATE( Catch::Generators::from_range( std::begin(k_nc_shapes), std::end(k_nc_shapes) ) );
	std::size_t const n = N * C;

	BatchNormLayer<float, Cpu> layer( 1e-5f, 0.1f );
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         std::vector<float>( n, 1.f ), std::vector<float>( n, 0.f ), N, C );
	layer.forward();

	REQUIRE( out_latch.shape()[0] == N );
	REQUIRE( out_latch.shape()[1] == C );
}

TEST_CASE( "BatchNormLayer2 rank2 CPU training forward normalizes to zero mean / unit std",
           "[bn1d2][rank2][cpu][forward][train]" )
{
	auto [N, C] = GENERATE( Catch::Generators::from_range( std::begin(k_nc_shapes), std::end(k_nc_shapes) ) );
	std::size_t const n = N * C;

	BatchNormLayer<float, Cpu> layer( 1e-5f, 0.1f );
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;

	// Non-trivial input: each element differs so columns have real variance.
	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = static_cast<float>( i + 1 );

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         in_data, std::vector<float>( n, 0.f ), N, C );
	layer.forward();

	float const* out = out_latch.fwdData();
	// Default gamma=1, beta=0 → each output column should have mean≈0, std≈1.
	for ( std::size_t c = 0; c < C; ++c ) {
		REQUIRE_THAT( col_mean( out, N, C, c ), WithinAbs( 0.f, eps_loose ) );
		REQUIRE_THAT( std::sqrt( col_var( out, N, C, c ) ), WithinAbs( 1.f, eps_loose ) );
	}
}

TEST_CASE( "BatchNormLayer2 rank2 CPU eval forward produces finite outputs",
           "[bn1d2][rank2][cpu][forward][eval]" )
{
	constexpr std::size_t N = 8, C = 2, n = N * C;

	BatchNormLayer<float, Cpu> layer( 1e-5f, 0.1f );
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = static_cast<float>( i );

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         in_data, std::vector<float>( n, 0.f ), N, C );

	layer.forward();           // training pass to build EMA stats
	layer.setTraining( false );
	layer.forward();           // eval pass

	float const* out = out_latch.fwdData();
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE( std::isfinite( out[i] ) );
}

TEST_CASE( "BatchNormLayer2 rank2 CPU backward grad_beta = sum(dy) per feature",
           "[bn1d2][rank2][cpu][backward][grad_beta]" )
{
	constexpr std::size_t N = 4, C = 3, n = N * C;

	BatchNormLayer<float, Cpu> layer( 1e-5f, 0.1f );
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = static_cast<float>( i + 1 );

	// All-ones upstream gradient.
	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         in_data, std::vector<float>( n, 1.f ), N, C );
	layer.forward();
	layer.backward();

	// grad_beta[c] = sum_n dy[n,c] = N (upstream grad is all-ones).
	float const* grad_beta = layer.getGradBias();
	for ( std::size_t c = 0; c < C; ++c )
		REQUIRE_THAT( grad_beta[c], WithinAbs( static_cast<float>( N ), eps_tight ) );
}

TEST_CASE( "BatchNormLayer2 rank2 CPU backward grad_gamma ≈ 0 for constant input",
           "[bn1d2][rank2][cpu][backward][grad_gamma]" )
{
	constexpr std::size_t N = 6, C = 2, n = N * C;

	BatchNormLayer<float, Cpu> layer( 1e-5f, 0.1f );
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;

	// Constant input per column → x_hat = 0 everywhere → grad_gamma = 0.
	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         std::vector<float>( n, 5.f ), std::vector<float>( n, 1.f ), N, C );
	layer.forward();
	layer.backward();

	float const* grad_gamma = layer.getGradWeights();
	for ( std::size_t c = 0; c < C; ++c )
		REQUIRE_THAT( grad_gamma[c], WithinAbs( 0.f, eps_tight ) );
}

TEST_CASE( "BatchNormLayer2 rank2 CPU backward dx sums to zero per feature",
           "[bn1d2][rank2][cpu][backward][dx]" )
{
	// BN backward: sum_n dx[n,c] = 0 when upstream grad is all-ones
	// (the (N*dy - sum_dy) term in the scale factor sums to zero over n).
	constexpr std::size_t N = 8, C = 3, n = N * C;

	BatchNormLayer<float, Cpu> layer( 1e-5f, 0.1f );
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = static_cast<float>( i * 2 + 1 );

	wireCpu( layer, in_latch, out_latch, g_in, g_out,
	         in_data, std::vector<float>( n, 1.f ), N, C );
	layer.forward();
	layer.backward();

	float const* dx = g_out.fwdData();
	for ( std::size_t c = 0; c < C; ++c ) {
		float col_sum = 0.f;
		for ( std::size_t ni = 0; ni < N; ++ni )
			col_sum += dx[ni * C + c];
		REQUIRE_THAT( col_sum, WithinAbs( 0.f, eps_loose ) );
	}
}

// ===========================================================================
// CPU tests — rank-4 NCHW latch
// ===========================================================================

TEST_CASE( "BatchNormLayer2 NCHW CPU forward output shape matches input",
           "[bn1d2][nchw][cpu][forward][shape]" )
{
	NchwShape const sh = GENERATE( Catch::Generators::from_range(
	    std::begin( k_nchw_shapes ), std::end( k_nchw_shapes ) ) );
	std::size_t const N = sh[0], C = sh[1], H = sh[2], W = sh[3];
	std::size_t const n_el = N * C * H * W;

	BatchNormLayer<float, Cpu> layer( 1e-5f, 0.1f );
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;

	wireCpuNchw( layer, in_latch, out_latch, g_in, g_out,
	             std::vector<float>( n_el, 1.f ), std::vector<float>( n_el, 0.f ), N, C, H, W );
	layer.forward();

	auto const &outs = out_latch.shape();
	REQUIRE( outs.size() == 4 );
	REQUIRE( outs[0] == N );
	REQUIRE( outs[1] == C );
	REQUIRE( outs[2] == H );
	REQUIRE( outs[3] == W );
}

TEST_CASE(
    "BatchNormLayer2 NCHW CPU training forward normalizes to zero mean / unit spatial variance per channel",
    "[bn1d2][nchw][cpu][forward][train]" )
{
	NchwShape const sh = GENERATE( Catch::Generators::from_range(
	    std::begin( k_nchw_shapes ), std::end( k_nchw_shapes ) ) );
	std::size_t const N = sh[0], C = sh[1], H = sh[2], W = sh[3];
	std::size_t const n_el = N * C * H * W;

	BatchNormLayer<float, Cpu> layer( 1e-5f, 0.1f );
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;

	std::vector<float> in_data( n_el );
	for ( std::size_t i = 0; i < n_el; ++i )
		in_data[i] = static_cast<float>( i + 1 );

	wireCpuNchw( layer, in_latch, out_latch, g_in, g_out,
	             in_data, std::vector<float>( n_el, 0.f ), N, C, H, W );
	layer.forward();

	float const* out = out_latch.fwdData();
	for ( std::size_t c = 0; c < C; ++c ) {
		REQUIRE_THAT( ch_mean_nchw( out, N, C, H, W, c ), WithinAbs( 0.f, eps_loose ) );
		REQUIRE_THAT( std::sqrt( ch_var_nchw( out, N, C, H, W, c ) ),
		              WithinAbs( 1.f, eps_loose ) );
	}
}

TEST_CASE( "BatchNormLayer2 NCHW CPU backward grad_beta = sum(dy) per channel spatially",
           "[bn1d2][nchw][cpu][backward][grad_beta]" )
{
	constexpr std::size_t N = 3, C = 2, H = 2, W = 2;
	std::size_t const M = N * H * W;
	std::size_t const n_el = N * C * H * W;

	BatchNormLayer<float, Cpu> layer( 1e-5f, 0.1f );
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;

	std::vector<float> in_data( n_el );
	for ( std::size_t i = 0; i < n_el; ++i )
		in_data[i] = static_cast<float>( i + 1 );

	wireCpuNchw( layer, in_latch, out_latch, g_in, g_out,
	             in_data, std::vector<float>( n_el, 1.f ), N, C, H, W );
	layer.forward();
	layer.backward();

	float const* grad_beta = layer.getGradBias();
	for ( std::size_t c = 0; c < C; ++c )
		REQUIRE_THAT( grad_beta[c], WithinAbs( static_cast<float>( M ), eps_loose ) );
}

TEST_CASE( "BatchNormLayer2 NCHW CPU backward dx sums to zero per channel across N×H×W",
           "[bn1d2][nchw][cpu][backward][dx]" )
{
	constexpr std::size_t N = 4, C = 3, H = 2, W = 2;
	std::size_t const n_el = N * C * H * W;

	BatchNormLayer<float, Cpu> layer( 1e-5f, 0.1f );
	LatchLayer<float, Cpu> in_latch, out_latch, g_in, g_out;

	std::vector<float> in_data( n_el );
	for ( std::size_t i = 0; i < n_el; ++i )
		in_data[i] = static_cast<float>( i ) * 0.25f + 1.f;

	wireCpuNchw( layer, in_latch, out_latch, g_in, g_out,
	             in_data, std::vector<float>( n_el, 1.f ), N, C, H, W );
	layer.forward();
	layer.backward();

	float const* dx = g_out.fwdData();
	for ( std::size_t c = 0; c < C; ++c )
		REQUIRE_THAT( dx_sum_ch_nchw( dx, N, C, H, W, c ), WithinAbs( 0.f, eps_loose ) );
}

// ===========================================================================
// CUDA tests
// ===========================================================================

#if NEURAL_CUDA_ENABLED

TEST_CASE( "BatchNormLayer2 rank2 CUDA forward matches CPU result",
           "[bn1d2][rank2][cuda][forward]" )
{
	constexpr std::size_t N = 8, C = 4, n = N * C;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = static_cast<float>( i + 1 );

	// CPU reference.
	BatchNormLayer<float, Cpu> cpu_layer( 1e-5f, 0.1f );
	LatchLayer<float, Cpu> cpu_in, cpu_out, cpu_gi, cpu_go;
	wireCpu( cpu_layer, cpu_in, cpu_out, cpu_gi, cpu_go,
	         in_data, std::vector<float>( n, 0.f ), N, C );
	cpu_layer.forward();

	// CUDA layer.
	BatchNormLayer<float, Cuda> cuda_layer( 1e-5f, 0.1f );
	LatchLayer<float, Cuda> cuda_in, cuda_out, cuda_gi, cuda_go;
	wireCuda( cuda_layer, cuda_in, cuda_out, cuda_gi, cuda_go,
	          in_data, std::vector<float>( n, 0.f ), N, C );
	cuda_layer.forward();
	cudaDeviceSynchronize();

	std::vector<float> cuda_result( n );
	cudaMemcpy( cuda_result.data(), cuda_out.fwdData(), n * sizeof(float), cudaMemcpyDeviceToHost );

	float const* cpu_result = cpu_out.fwdData();
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( cuda_result[i], WithinAbs( cpu_result[i], eps_loose ) );
}

TEST_CASE( "BatchNormLayer2 rank2 CUDA backward matches CPU gradients",
           "[bn1d2][rank2][cuda][backward]" )
{
	constexpr std::size_t N = 6, C = 3, n = N * C;

	std::vector<float> in_data( n ), g_in_data( n );
	for ( std::size_t i = 0; i < n; ++i ) {
		in_data[i]   = static_cast<float>( i ) * 0.5f + 1.f;
		g_in_data[i] = static_cast<float>( i % 3 + 1 );
	}

	// CPU reference.
	BatchNormLayer<float, Cpu> cpu_layer( 1e-5f, 0.1f );
	LatchLayer<float, Cpu> cpu_in, cpu_out, cpu_gi, cpu_go;
	wireCpu( cpu_layer, cpu_in, cpu_out, cpu_gi, cpu_go, in_data, g_in_data, N, C );
	cpu_layer.forward();
	cpu_layer.backward();

	// CUDA layer.
	BatchNormLayer<float, Cuda> cuda_layer( 1e-5f, 0.1f );
	LatchLayer<float, Cuda> cuda_in, cuda_out, cuda_gi, cuda_go;
	wireCuda( cuda_layer, cuda_in, cuda_out, cuda_gi, cuda_go, in_data, g_in_data, N, C );
	cuda_layer.forward();
	cuda_layer.backward();
	cudaDeviceSynchronize();

	// Compare dx.
	std::vector<float> cuda_dx( n );
	cudaMemcpy( cuda_dx.data(), cuda_go.fwdData(), n * sizeof(float), cudaMemcpyDeviceToHost );
	float const* cpu_dx = cpu_go.fwdData();
	for ( std::size_t i = 0; i < n; ++i )
		REQUIRE_THAT( cuda_dx[i], WithinAbs( cpu_dx[i], eps_loose ) );

	// Compare grad_gamma and grad_beta.
	std::vector<float> cuda_dg( C ), cuda_db( C );
	cudaMemcpy( cuda_dg.data(), cuda_layer.getGradWeights(), C * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( cuda_db.data(), cuda_layer.getGradBias(),    C * sizeof(float), cudaMemcpyDeviceToHost );
	float const* cpu_dg = cpu_layer.getGradWeights();
	float const* cpu_db = cpu_layer.getGradBias();
	for ( std::size_t c = 0; c < C; ++c ) {
		REQUIRE_THAT( cuda_dg[c], WithinAbs( cpu_dg[c], eps_loose ) );
		REQUIRE_THAT( cuda_db[c], WithinAbs( cpu_db[c], eps_loose ) );
	}
}

TEST_CASE( "BatchNormLayer2 rank2 CUDA eval forward produces finite outputs",
           "[bn1d2][rank2][cuda][forward][eval]" )
{
	constexpr std::size_t N = 8, C = 3, n = N * C;

	std::vector<float> in_data( n );
	for ( std::size_t i = 0; i < n; ++i )
		in_data[i] = static_cast<float>( i + 1 );

	BatchNormLayer<float, Cuda> layer( 1e-5f, 0.1f );
	LatchLayer<float, Cuda> in_latch, out_latch, g_in, g_out;
	wireCuda( layer, in_latch, out_latch, g_in, g_out,
	          in_data, std::vector<float>( n, 0.f ), N, C );

	layer.forward();           // training pass to build EMA stats
	layer.setTraining( false );
	layer.forward();           // eval pass
	cudaDeviceSynchronize();

	std::vector<float> out( n );
	cudaMemcpy( out.data(), out_latch.fwdData(), n * sizeof(float), cudaMemcpyDeviceToHost );
	for ( float v : out ) {
		REQUIRE( std::isfinite( v ) );
		REQUIRE_THAT( v, WithinAbs( 0.f, 10.f ) );
	}
}

TEST_CASE( "BatchNormLayer2 NCHW CUDA forward matches CPU result", "[bn1d2][nchw][cuda][forward]" )
{
	constexpr std::size_t N = 4, C = 3, H = 2, W = 3;
	std::size_t const n_el = N * C * H * W;

	std::vector<float> in_data( n_el );
	for ( std::size_t i = 0; i < n_el; ++i )
		in_data[i] = static_cast<float>( i + 1 ) * 0.1f;

	BatchNormLayer<float, Cpu> cpu_layer( 1e-5f, 0.1f );
	LatchLayer<float, Cpu> cpu_in, cpu_out, cpu_gi, cpu_go;
	wireCpuNchw( cpu_layer, cpu_in, cpu_out, cpu_gi, cpu_go,
	             in_data, std::vector<float>( n_el, 0.f ), N, C, H, W );
	cpu_layer.forward();

	BatchNormLayer<float, Cuda> cuda_layer( 1e-5f, 0.1f );
	LatchLayer<float, Cuda> cuda_in, cuda_out, cuda_gi, cuda_go;
	wireCudaNchw( cuda_layer, cuda_in, cuda_out, cuda_gi, cuda_go,
	              in_data, std::vector<float>( n_el, 0.f ), N, C, H, W );
	cuda_layer.forward();
	cudaDeviceSynchronize();

	std::vector<float> cuda_result( n_el );
	cudaMemcpy( cuda_result.data(), cuda_out.fwdData(), n_el * sizeof(float), cudaMemcpyDeviceToHost );

	float const* cpu_result = cpu_out.fwdData();
	for ( std::size_t i = 0; i < n_el; ++i )
		REQUIRE_THAT( cuda_result[i], WithinAbs( cpu_result[i], eps_loose ) );
}

TEST_CASE( "BatchNormLayer2 NCHW CUDA backward matches CPU gradients", "[bn1d2][nchw][cuda][backward]" )
{
	constexpr std::size_t N = 3, C = 2, H = 2, W = 2;
	std::size_t const n_el = N * C * H * W;

	std::vector<float> in_data( n_el ), g_in_data( n_el );
	for ( std::size_t i = 0; i < n_el; ++i ) {
		in_data[i]   = static_cast<float>( i ) * 0.3f + 0.5f;
		g_in_data[i] = static_cast<float>( ( i + 1 ) % 5 ) * 0.2f;
	}

	BatchNormLayer<float, Cpu> cpu_layer( 1e-5f, 0.1f );
	LatchLayer<float, Cpu> cpu_in, cpu_out, cpu_gi, cpu_go;
	wireCpuNchw( cpu_layer, cpu_in, cpu_out, cpu_gi, cpu_go, in_data, g_in_data, N, C, H, W );
	cpu_layer.forward();
	cpu_layer.backward();

	BatchNormLayer<float, Cuda> cuda_layer( 1e-5f, 0.1f );
	LatchLayer<float, Cuda> cuda_in, cuda_out, cuda_gi, cuda_go;
	wireCudaNchw( cuda_layer, cuda_in, cuda_out, cuda_gi, cuda_go, in_data, g_in_data, N, C, H, W );
	cuda_layer.forward();
	cuda_layer.backward();
	cudaDeviceSynchronize();

	std::vector<float> cuda_dx( n_el );
	cudaMemcpy( cuda_dx.data(), cuda_go.fwdData(), n_el * sizeof(float), cudaMemcpyDeviceToHost );
	float const* cpu_dx = cpu_go.fwdData();
	for ( std::size_t i = 0; i < n_el; ++i )
		REQUIRE_THAT( cuda_dx[i], WithinAbs( cpu_dx[i], eps_loose ) );

	std::vector<float> cuda_dg( C ), cuda_db( C );
	cudaMemcpy( cuda_dg.data(), cuda_layer.getGradWeights(), C * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( cuda_db.data(), cuda_layer.getGradBias(), C * sizeof(float), cudaMemcpyDeviceToHost );
	float const* cpu_dg = cpu_layer.getGradWeights();
	float const* cpu_db = cpu_layer.getGradBias();
	for ( std::size_t c = 0; c < C; ++c ) {
		REQUIRE_THAT( cuda_dg[c], WithinAbs( cpu_dg[c], eps_loose ) );
		REQUIRE_THAT( cuda_db[c], WithinAbs( cpu_db[c], eps_loose ) );
	}
}

#endif // NEURAL_CUDA_ENABLED
