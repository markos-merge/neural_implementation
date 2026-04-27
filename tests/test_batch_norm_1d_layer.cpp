#include "batch_norm_1d_layer.hpp"
#include "layer_wiring_helpers.hpp"
#include "tensor.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

#if NEURAL_CUDNN_ENABLED
#include "cuda_tensor.hpp"
#include "neural_cuda_runtime.hpp"
#endif

using neural::BatchNorm1dLayer;
using neural::Tensor;
using neural::test::wire_layer;
using Catch::Matchers::WithinAbs;

namespace {
constexpr float eps_tight = 1e-5f;
constexpr float eps_loose = 1e-3f;

/// Compute mean over rows of column c from a flat row-major buffer (N rows, C cols).
float col_mean( float const *data, std::size_t N, std::size_t C, std::size_t c )
{
	float s = 0.f;
	for ( std::size_t n = 0; n < N; ++n )
		s += data[n * C + c];
	return s / static_cast<float>( N );
}

/// Compute variance over rows of column c from a flat row-major buffer.
float col_var( float const *data, std::size_t N, std::size_t C, std::size_t c )
{
	float const mu = col_mean( data, N, C, c );
	float v        = 0.f;
	for ( std::size_t n = 0; n < N; ++n ) {
		float const d = data[n * C + c] - mu;
		v += d * d;
	}
	return v / static_cast<float>( N );
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// CPU (Tensor<float>) tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE( "BatchNorm1dLayer CPU training forward normalizes to zero mean / unit std",
           "[batch_norm_1d][cpu][forward][train]" )
{
	constexpr std::size_t N = 16, C = 4;
	BatchNorm1dLayer<Tensor<float>> layer( C, 1e-5f, 0.1f );

	// Non-trivial input: each column has a different mean and variance.
	Tensor<float> input( N, C );
	for ( std::size_t n = 0; n < N; ++n )
		for ( std::size_t c = 0; c < C; ++c )
			input.assign( n, c, static_cast<float>( n * C + c + 1 ) );

	Tensor<float> output( N, C );
	Tensor<float> g_out( N, C, 0.f );
	Tensor<float> g_in( N, C );
	wire_layer( layer, input, output, g_out, g_in );
	layer.forward();

	// With gamma = 1, beta = 0: each output column should have mean ≈ 0, std ≈ 1.
	for ( std::size_t c = 0; c < C; ++c ) {
		float const mean = col_mean( output.data(), N, C, c );
		float const var  = col_var( output.data(), N, C, c );
		REQUIRE_THAT( mean, WithinAbs( 0.f, eps_loose ) );
		REQUIRE_THAT( std::sqrt( var ), WithinAbs( 1.f, eps_loose ) );
	}
}

TEST_CASE( "BatchNorm1dLayer CPU eval forward uses running stats",
           "[batch_norm_1d][cpu][forward][eval]" )
{
	constexpr std::size_t N = 8, C = 2;
	BatchNorm1dLayer<Tensor<float>> layer( C, 1e-5f, 0.1f );

	Tensor<float> input( N, C );
	for ( std::size_t n = 0; n < N; ++n ) {
		input.assign( n, 0, static_cast<float>( n ) );
		input.assign( n, 1, static_cast<float>( n ) * 2.f );
	}
	Tensor<float> output( N, C );
	Tensor<float> g_out( N, C, 0.f );
	Tensor<float> g_in( N, C );
	wire_layer( layer, input, output, g_out, g_in );

	// One training pass to build EMA stats.
	layer.forward();

	// Switch to eval: output should still produce finite, non-NaN values.
	layer.setTraining( false );
	layer.forward();
	for ( std::size_t n = 0; n < N; ++n )
		for ( std::size_t c = 0; c < C; ++c )
			REQUIRE( std::isfinite( output( n, c ) ) );
}

TEST_CASE( "BatchNorm1dLayer CPU backward grad_beta = sum(dL/dy) per feature",
           "[batch_norm_1d][cpu][backward][grad_beta]" )
{
	constexpr std::size_t N = 4, C = 3;
	BatchNorm1dLayer<Tensor<float>> layer( C, 1e-5f, 0.1f );

	Tensor<float> input( N, C );
	for ( std::size_t n = 0; n < N; ++n )
		for ( std::size_t c = 0; c < C; ++c )
			input.assign( n, c, static_cast<float>( n + c + 1 ) );

	Tensor<float> output( N, C );
	Tensor<float> g_out( N, C );
	Tensor<float> g_in( N, C, 0.f );
	wire_layer( layer, input, output, g_out, g_in );
	layer.forward();

	// Upstream gradient: all ones.
	for ( std::size_t n = 0; n < N; ++n )
		for ( std::size_t c = 0; c < C; ++c )
			g_out.assign( n, c, 1.f );

	layer.backward();

	// grad_beta[c] = sum_n d_output[n,c] = N (all-ones upstream grad).
	for ( std::size_t c = 0; c < C; ++c )
		REQUIRE_THAT( layer.getGradBias()( 0, c ),
		              WithinAbs( static_cast<float>( N ), eps_tight ) );
}

TEST_CASE( "BatchNorm1dLayer CPU backward grad_gamma = sum(dy * x_hat) per feature",
           "[batch_norm_1d][cpu][backward][grad_gamma]" )
{
	constexpr std::size_t N = 6, C = 2;
	BatchNorm1dLayer<Tensor<float>> layer( C, 1e-5f, 0.1f );

	// Use constant input per column so x_hat = 0 everywhere → grad_gamma = 0.
	Tensor<float> input( N, C, 5.f );
	Tensor<float> output( N, C );
	Tensor<float> g_out( N, C, 1.f );
	Tensor<float> g_in( N, C, 0.f );
	wire_layer( layer, input, output, g_out, g_in );
	layer.forward();
	layer.backward();

	// x_hat = (5 - 5) / sqrt(0 + eps) = 0 → grad_gamma = sum(1 * 0) = 0.
	for ( std::size_t c = 0; c < C; ++c )
		REQUIRE_THAT( layer.getGradWeights()( 0, c ), WithinAbs( 0.f, eps_tight ) );
}

TEST_CASE( "BatchNorm1dLayer CPU backward dx sums to zero per feature",
           "[batch_norm_1d][cpu][backward][dx]" )
{
	// For batch norm, sum_n dx[n,c] = 0 when upstream grad is all-ones
	// (because BN removes the mean from the normalized path and the scale term
	//  contains a factor (N * dy - sum_dy) which sums to zero over n).
	constexpr std::size_t N = 8, C = 3;
	BatchNorm1dLayer<Tensor<float>> layer( C, 1e-5f, 0.1f );

	Tensor<float> input( N, C );
	for ( std::size_t n = 0; n < N; ++n )
		for ( std::size_t c = 0; c < C; ++c )
			input.assign( n, c, static_cast<float>( n * 2 + c ) );

	Tensor<float> output( N, C );
	Tensor<float> g_out( N, C, 1.f );
	Tensor<float> g_in( N, C, 0.f );
	wire_layer( layer, input, output, g_out, g_in );
	layer.forward();
	layer.backward();

	for ( std::size_t c = 0; c < C; ++c ) {
		float col_sum = 0.f;
		for ( std::size_t n = 0; n < N; ++n )
			col_sum += g_in( n, c );
		REQUIRE_THAT( col_sum, WithinAbs( 0.f, eps_loose ) );
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// CUDA (CudaTensor<float>) tests — only compiled and run when cuDNN is available
// ─────────────────────────────────────────────────────────────────────────────

#if NEURAL_CUDNN_ENABLED

using neural::CudaTensor;

namespace {

/// Copy a CudaTensor<float> to a host std::vector.
std::vector<float> to_host( CudaTensor<float> const &t )
{
	std::vector<float> h( t.size() );
	t.copyToHost( h.data() );
	return h;
}

/// Populate a CudaTensor<float> from a flat host vector.
void from_host( CudaTensor<float> &t, std::vector<float> const &v )
{
	t.assign( v.data(), v.size() );
}

} // namespace

TEST_CASE( "BatchNorm1dLayer CUDA forward matches CPU result",
           "[batch_norm_1d][cuda][forward]" )
{
	if ( !neural::cuda_runtime_ready() )
		SKIP( "No CUDA device available" );

	constexpr std::size_t N = 8, C = 4;

	// Build a non-trivial input and fill both CPU and CUDA tensors identically.
	std::vector<float> input_vals( N * C );
	for ( std::size_t i = 0; i < input_vals.size(); ++i )
		input_vals[i] = static_cast<float>( i + 1 );

	// CPU layer.
	BatchNorm1dLayer<Tensor<float>> cpu_layer( C, 1e-5f, 0.1f );
	Tensor<float> cpu_in( N, C, input_vals.begin(), input_vals.end() );
	Tensor<float> cpu_out( N, C );
	Tensor<float> cpu_g_out( N, C, 0.f );
	Tensor<float> cpu_g_in( N, C );
	wire_layer( cpu_layer, cpu_in, cpu_out, cpu_g_out, cpu_g_in );
	cpu_layer.forward();

	// CUDA layer.
	BatchNorm1dLayer<CudaTensor<float>> cuda_layer( C, 1e-5f, 0.1f );
	CudaTensor<float> cuda_in( N, C );
	from_host( cuda_in, input_vals );
	CudaTensor<float> cuda_out( N, C, 0.f );
	CudaTensor<float> cuda_g_out( N, C, 0.f );
	CudaTensor<float> cuda_g_in( N, C, 0.f );
	wire_layer( cuda_layer, cuda_in, cuda_out, cuda_g_out, cuda_g_in );
	cuda_layer.forward();

	std::vector<float> cuda_result = to_host( cuda_out );
	for ( std::size_t n = 0; n < N; ++n ) {
		for ( std::size_t c = 0; c < C; ++c ) {
			std::size_t const i = n * C + c;
			REQUIRE_THAT( cuda_result[i], WithinAbs( cpu_out( n, c ), eps_loose ) );
		}
	}
}

TEST_CASE( "BatchNorm1dLayer CUDA backward matches CPU gradients",
           "[batch_norm_1d][cuda][backward]" )
{
	if ( !neural::cuda_runtime_ready() )
		SKIP( "No CUDA device available" );

	constexpr std::size_t N = 6, C = 3;

	std::vector<float> input_vals( N * C );
	std::vector<float> g_out_vals( N * C );
	for ( std::size_t i = 0; i < N * C; ++i ) {
		input_vals[i] = static_cast<float>( i ) * 0.5f + 1.f;
		g_out_vals[i] = static_cast<float>( i % 3 + 1 );
	}

	// CPU.
	BatchNorm1dLayer<Tensor<float>> cpu_layer( C, 1e-5f, 0.1f );
	Tensor<float> cpu_in( N, C, input_vals.begin(), input_vals.end() );
	Tensor<float> cpu_out( N, C );
	Tensor<float> cpu_g_out( N, C, g_out_vals.begin(), g_out_vals.end() );
	Tensor<float> cpu_g_in( N, C, 0.f );
	wire_layer( cpu_layer, cpu_in, cpu_out, cpu_g_out, cpu_g_in );
	cpu_layer.forward();
	cpu_layer.backward();

	// CUDA.
	BatchNorm1dLayer<CudaTensor<float>> cuda_layer( C, 1e-5f, 0.1f );
	CudaTensor<float> cuda_in( N, C );
	from_host( cuda_in, input_vals );
	CudaTensor<float> cuda_out( N, C, 0.f );
	CudaTensor<float> cuda_g_out( N, C );
	from_host( cuda_g_out, g_out_vals );
	CudaTensor<float> cuda_g_in( N, C, 0.f );
	wire_layer( cuda_layer, cuda_in, cuda_out, cuda_g_out, cuda_g_in );
	cuda_layer.forward();
	cuda_layer.backward();

	// Compare dx.
	std::vector<float> cuda_dx = to_host( cuda_g_in );
	for ( std::size_t n = 0; n < N; ++n ) {
		for ( std::size_t c = 0; c < C; ++c ) {
			std::size_t const i = n * C + c;
			REQUIRE_THAT( cuda_dx[i], WithinAbs( cpu_g_in( n, c ), eps_loose ) );
		}
	}

	// Compare grad_gamma and grad_beta.
	std::vector<float> cuda_dg = to_host( cuda_layer.getGradWeights() );
	std::vector<float> cuda_db = to_host( cuda_layer.getGradBias() );
	for ( std::size_t c = 0; c < C; ++c ) {
		REQUIRE_THAT( cuda_dg[c], WithinAbs( cpu_layer.getGradWeights()( 0, c ), eps_loose ) );
		REQUIRE_THAT( cuda_db[c], WithinAbs( cpu_layer.getGradBias()( 0, c ), eps_loose ) );
	}
}

TEST_CASE( "BatchNorm1dLayer CUDA eval forward produces finite normalized outputs",
           "[batch_norm_1d][cuda][forward][eval]" )
{
	// CPU vs cuDNN EMA for running variance can differ; we do not compare them here.
	// After a *single* training forward, the EMA for mean/variance is only partially updated
	// (momentum 0.1 from init). Running the same batch again in eval uses those incomplete
	// EMAs, so the activations are *not* guaranteed to be tight z-scores; |y| can slightly
	// exceed 6 even when everything is correct. (Observed: ~6.28.) We only require finite
	// values and a loose magnitude bound, not "within [-6,6]".
	if ( !neural::cuda_runtime_ready() )
		SKIP( "No CUDA device available" );

	constexpr std::size_t N = 8, C = 3;
	std::vector<float> input_vals( N * C );
	for ( std::size_t i = 0; i < N * C; ++i )
		input_vals[i] = static_cast<float>( i + 1 );

	BatchNorm1dLayer<CudaTensor<float>> cuda_layer( C, 1e-5f, 0.1f );
	CudaTensor<float> cuda_in( N, C );
	from_host( cuda_in, input_vals );
	CudaTensor<float> cuda_out( N, C, 0.f );
	CudaTensor<float> cuda_g_out( N, C, 0.f );
	CudaTensor<float> cuda_g_in( N, C, 0.f );
	wire_layer( cuda_layer, cuda_in, cuda_out, cuda_g_out, cuda_g_in );

	// Build EMA running stats with one training pass.
	cuda_layer.forward();

	// Eval pass should produce finite values normalised to a reasonable range.
	cuda_layer.setTraining( false );
	cuda_layer.forward();

	std::vector<float> result = to_host( cuda_out );
	for ( float v : result ) {
		REQUIRE( std::isfinite( v ) );
		// Single-step EMA + eval on same data can still yield |y| > 6; keep a loose sanity bound.
		REQUIRE_THAT( v, WithinAbs( 0.f, 10.f ) );
	}
}

#endif // NEURAL_CUDNN_ENABLED
