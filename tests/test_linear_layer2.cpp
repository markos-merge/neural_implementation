#include "refactor/latch_layer.hpp"
#include "refactor/linear_layer.hpp"
#include "tensor.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cstring>
#include <utility>
#include <vector>
#if NEURAL_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

using neural::refactor::Cpu;
using neural::refactor::LatchLayer;
using neural::refactor::LinearLayer;
using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps = 1e-5f;

using SizePair = std::pair<std::size_t, std::size_t>;

static const SizePair k_shapes[] = {
    { 1, 2 }, { 2, 3 }, { 7, 5 }, { 16, 64 }, { 4, 16 },
};

// ---- CPU wiring -------------------------------------------------------------

void wireCpu(
    LinearLayer<float, Cpu>&  layer,
    LatchLayer<float, Cpu>&   in_latch,
    LatchLayer<float, Cpu>&   out_latch,
    LatchLayer<float, Cpu>&   g_upstream_latch,
    LatchLayer<float, Cpu>&   g_prior_latch,
    std::vector<float> const &in_data,
    std::vector<float> const &upstream_grad_data,
    std::size_t rows,
    std::size_t in_cols,
    std::size_t out_features )
{
	in_latch.resize( { rows, in_cols } );
	std::memcpy( in_latch.fwdData(), in_data.data(), in_data.size() * sizeof( float ) );

	out_latch.resize( { rows, out_features } );

	g_upstream_latch.resize( { rows, out_features } );
	std::memcpy( g_upstream_latch.fwdData(), upstream_grad_data.data(),
	            upstream_grad_data.size() * sizeof( float ) );

	g_prior_latch.resize( { rows, in_cols } );

	layer.setInputOutputLatches( &in_latch, &out_latch );
	layer.setGradLatches( &g_upstream_latch, &g_prior_latch );
}

/// After \c forward() allocates parameters, overwrite weights and bias with known values
/// (row-major: \c weight rows = in_features, cols = out_features).
void setCpuKnownWeights(
    LinearLayer<float, Cpu> &layer,
    std::vector<float> const &w_rowmajor,
    std::vector<float> const &bias_col /* out_features x 1 */ )
{
	float *w = layer.getWeights();
	float *b = layer.getBias();
	REQUIRE( w != nullptr );
	REQUIRE( b != nullptr );
	std::memcpy( w, w_rowmajor.data(), w_rowmajor.size() * sizeof( float ) );
	std::memcpy( b, bias_col.data(), bias_col.size() * sizeof( float ) );
}

#if NEURAL_CUDA_ENABLED
using neural::refactor::Cuda;

void wireCuda(
    LinearLayer<float, Cuda> & layer,
    LatchLayer<float, Cuda> &  in_latch,
    LatchLayer<float, Cuda> &  out_latch,
    LatchLayer<float, Cuda> &  g_upstream_latch,
    LatchLayer<float, Cuda> &  g_prior_latch,
    std::vector<float> const &in_data,
    std::vector<float> const &upstream_grad_data,
    std::size_t rows,
    std::size_t in_cols,
    std::size_t out_features )
{
	in_latch.resize( { rows, in_cols } );
	cudaMemcpy( in_latch.fwdData(), in_data.data(), in_data.size() * sizeof( float ),
	            cudaMemcpyHostToDevice );

	out_latch.resize( { rows, out_features } );

	g_upstream_latch.resize( { rows, out_features } );
	cudaMemcpy( g_upstream_latch.fwdData(), upstream_grad_data.data(),
	            upstream_grad_data.size() * sizeof( float ), cudaMemcpyHostToDevice );

	g_prior_latch.resize( { rows, in_cols } );

	layer.setInputOutputLatches( &in_latch, &out_latch );
	layer.setGradLatches( &g_upstream_latch, &g_prior_latch );
}

void setCudaKnownWeights(
    LinearLayer<float, Cuda> &layer,
    std::vector<float> const &w_rowmajor,
    std::vector<float> const &bias_col )
{
	std::size_t const nw = w_rowmajor.size();
	std::size_t const nb = bias_col.size();
	cudaMemcpy( layer.getWeights(), w_rowmajor.data(), nw * sizeof( float ),
	            cudaMemcpyHostToDevice );
	cudaMemcpy( layer.getBias(), bias_col.data(), nb * sizeof( float ),
	            cudaMemcpyHostToDevice );
	cudaDeviceSynchronize();
}
#endif

} // namespace

// ============================================================================
// CPU tests
// ============================================================================

TEST_CASE( "LinearLayer2 CPU forward output shape", "[linear2][cpu][forward][shape]" )
{
	auto const [rows, in_cols] =
	    GENERATE( Catch::Generators::from_range( std::begin( k_shapes ), std::end( k_shapes ) ) );
	std::size_t const out_features = 7;

	LinearLayer<float, Cpu> layer( out_features );
	LatchLayer<float, Cpu> in_latch, out_latch, g_up, g_prior;
	std::size_t const n_in  = rows * in_cols;
	std::size_t const n_out = rows * out_features;

	wireCpu( layer, in_latch, out_latch, g_up, g_prior, std::vector<float>( n_in, 1.f ),
	         std::vector<float>( n_out, 0.f ), rows, in_cols, out_features );
	layer.forward();

	REQUIRE( out_latch.shape().size() == 2u );
	REQUIRE( out_latch.shape()[0] == rows );
	REQUIRE( out_latch.shape()[1] == out_features );
	REQUIRE( layer.outFeatures() == out_features );
}

TEST_CASE( "LinearLayer2 CPU forward with known weights matches matmul reference", "[linear2][cpu][forward][numerical]" )
{
	std::size_t const rows         = 1;
	std::size_t const in_features = 2;
	std::size_t const out_features = 2;

	std::vector<float> const w_data = { 0.5f, 0.3f, 0.2f, 0.4f }; // 2x2 row-major
	std::vector<float> const b_data = { 0.1f, 0.2f };
	std::vector<float> const x_data = { 1.0f, 2.0f };

	LinearLayer<float, Cpu> layer( out_features );
	LatchLayer<float, Cpu> in_latch, out_latch, g_up, g_prior;
	wireCpu( layer, in_latch, out_latch, g_up, g_prior, x_data,
	         std::vector<float>( rows * out_features, 0.f ), rows, in_features, out_features );
	layer.forward();
	setCpuKnownWeights( layer, w_data, b_data );
	layer.forward();

	float const *out = out_latch.fwdData();
	REQUIRE_THAT( out[0], WithinAbs( 1.0f, eps ) );  // 1*0.5+2*0.2+0.1
	REQUIRE_THAT( out[1], WithinAbs( 1.3f, eps ) );  // 1*0.3+2*0.4+0.2
}

TEST_CASE( "LinearLayer2 CPU backward gradient shapes", "[linear2][cpu][backward][shape]" )
{
	std::size_t const rows         = 2;
	std::size_t const in_features = 3;
	std::size_t const out_features = 5;

	LinearLayer<float, Cpu> layer( out_features );
	LatchLayer<float, Cpu> in_latch, out_latch, g_up, g_prior;
	std::vector<float> x( rows * in_features, 1.f );
	std::vector<float> gup( rows * out_features, 1.f );
	wireCpu( layer, in_latch, out_latch, g_up, g_prior, x, gup, rows, in_features,
	         out_features );

	layer.forward();
	layer.backward();

	REQUIRE( g_prior.shape()[0] == rows );
	REQUIRE( g_prior.shape()[1] == in_features );

	REQUIRE( layer.getGradWeights() != nullptr );
	REQUIRE( layer.getGradBias() != nullptr );
	REQUIRE( layer.getWeights() != nullptr );
	// Internal tensor layout: weights (in_features x out_features)
	neural::Tensor<float> gw_view( layer.getGradWeights(), in_features, out_features, false );
	REQUIRE( gw_view.rows() == in_features );
	REQUIRE( gw_view.cols() == out_features );
	neural::Tensor<float> gb_view( layer.getGradBias(), out_features, 1, false );
	REQUIRE( gb_view.rows() == out_features );
	REQUIRE( gb_view.cols() == 1u );
}

TEST_CASE( "LinearLayer2 CPU backward numerical matches analytic", "[linear2][cpu][backward][numerical]" )
{
	std::size_t const rows         = 1;
	std::size_t const in_features = 2;
	std::size_t const out_features = 1;

	std::vector<float> const w_data = { 0.6f, 0.7f };
	std::vector<float> const b_data = { 0.1f };
	std::vector<float> const x_data = { 1.0f, 1.3f };
	float const upstream           = -0.39f;

	LinearLayer<float, Cpu> layer( out_features );
	LatchLayer<float, Cpu> in_latch, out_latch, g_up, g_prior;
	wireCpu( layer, in_latch, out_latch, g_up, g_prior, x_data, { upstream }, rows,
	         in_features, out_features );
	layer.forward();
	setCpuKnownWeights( layer, w_data, b_data );
	layer.forward();
	std::memcpy( g_up.fwdData(), &upstream, sizeof( float ) );
	layer.backward();

	float const *gp = g_prior.fwdData();
	REQUIRE_THAT( gp[0], WithinAbs( -0.234f, eps ) );  // -0.39 * 0.6
	REQUIRE_THAT( gp[1], WithinAbs( -0.273f, eps ) );  // -0.39 * 0.7

	neural::Tensor<float> gw( layer.getGradWeights(), in_features, out_features, false );
	neural::Tensor<float> gb( layer.getGradBias(), out_features, 1, false );
	REQUIRE_THAT( gw( 0, 0 ), WithinAbs( -0.39f, eps ) );   // 1 * -0.39
	REQUIRE_THAT( gw( 1, 0 ), WithinAbs( -0.507f, eps ) );  // 1.3 * -0.39
	REQUIRE_THAT( gb( 0, 0 ), WithinAbs( -0.39f, eps ) );
}

TEST_CASE( "LinearLayer2 CPU backward without forward throws", "[linear2][cpu][backward][error]" )
{
	LinearLayer<float, Cpu> layer( 3 );
	LatchLayer<float, Cpu> in_latch, out_latch, g_up, g_prior;
	wireCpu( layer, in_latch, out_latch, g_up, g_prior, std::vector<float>( 4, 0.f ),
	         std::vector<float>( 6, 1.f ), 2, 2, 3 );

	REQUIRE_THROWS_AS( layer.backward(), std::logic_error );
}

TEST_CASE( "LinearLayer2 CPU mismatched upstream gradient shape throws after forward", "[linear2][cpu][backward][error]" )
{
	LinearLayer<float, Cpu> layer( 2 );
	LatchLayer<float, Cpu> in_latch, out_latch, g_up, g_prior;
	// in 2x2 -> out should be 2x2; corrupt upstream to wrong width by resizing grad latch only
	in_latch.resize( { 2, 2 } );
	out_latch.resize( { 2, 2 } );
	g_up.resize( { 2, 999 } ); // wrong
	g_prior.resize( { 2, 2 } );
	layer.setInputOutputLatches( &in_latch, &out_latch );
	layer.setGradLatches( &g_up, &g_prior );
	layer.forward();

	REQUIRE_THROWS_AS( layer.backward(), std::logic_error );
}

TEST_CASE( "LinearLayer2 CPU rank-3 latch forward shape", "[linear2][cpu][forward][nd]" )
{
	std::vector<std::size_t> const shape       = { 2, 2, 3 };
	std::size_t const             flat         = 2 * 2 * 3;
	std::size_t const             out_features = 4;

	std::vector<float> in_data( flat );
	for ( std::size_t i = 0; i < flat; ++i )
		in_data[i] = static_cast<float>( i ) * 0.05f;

	LinearLayer<float, Cpu> layer( out_features );
	LatchLayer<float, Cpu> in_latch, out_latch, g_up, g_prior;
	in_latch.resize( shape );
	std::memcpy( in_latch.fwdData(), in_data.data(), flat * sizeof( float ) );
	out_latch.resize( { 2u, out_features } );
	g_up.resize( { 2u, out_features } );
	g_prior.resize( shape );
	layer.setInputOutputLatches( &in_latch, &out_latch );
	layer.setGradLatches( &g_up, &g_prior );

	layer.forward();

	REQUIRE( out_latch.shape()[0] == 2u );
	REQUIRE( out_latch.shape()[1] == out_features );
}

TEST_CASE( "LinearLayer2 CPU gradient check on bias with fixed weights", "[linear2][cpu][gradient_check]" )
{
	constexpr float       h            = 1e-4f;
	std::size_t const     rows         = 1;
	std::size_t const     in_features = 2;
	std::size_t const     out_features = 2;
	std::vector<float> const w_fixed = { 0.5f, 0.25f, -0.1f, 0.3f };
	std::vector<float> b0            = { 0.05f, -0.07f };
	std::vector<float> const x_data  = { 0.8f, -1.2f };
	std::vector<float> const ones_grad( rows * out_features, 1.f );

	LinearLayer<float, Cpu> layer( out_features );
	LatchLayer<float, Cpu> in_latch, out_latch, g_up, g_prior;
	wireCpu( layer, in_latch, out_latch, g_up, g_prior, x_data, ones_grad, rows,
	         in_features, out_features );
	layer.forward();
	setCpuKnownWeights( layer, w_fixed, b0 );
	layer.forward();
	std::memcpy( g_up.fwdData(), ones_grad.data(), ones_grad.size() * sizeof( float ) );
	layer.backward();

	neural::Tensor<float> gb_analytic( layer.getGradBias(), out_features, 1, false );

	auto bias_numeric_component = [&]( std::size_t idx ) {
		auto central_diff = [&]( float delta ) {
			std::vector<float> b = b0;
			b[idx] += delta;
			setCpuKnownWeights( layer, w_fixed, b );
			layer.forward();
			float const *yo = out_latch.fwdData();
			float sum = 0.f;
			for ( std::size_t k = 0; k < rows * out_features; ++k )
				sum += yo[k] * ones_grad[k];
			return sum;
		};
		return ( central_diff( h ) - central_diff( -h ) ) / ( 2.f * h );
	};

	for ( std::size_t j = 0; j < out_features; ++j ) {
		float const ng = bias_numeric_component( j );
		REQUIRE_THAT( gb_analytic( j, 0 ), WithinAbs( ng, 5e-3f ) );
	}
}

// ============================================================================
// CUDA tests
// ============================================================================

#if NEURAL_CUDA_ENABLED

TEST_CASE( "LinearLayer2 CUDA forward output shape", "[linear2][cuda][forward][shape]" )
{
	auto const [rows, in_cols] =
	    GENERATE( Catch::Generators::from_range( std::begin( k_shapes ), std::end( k_shapes ) ) );
	std::size_t const out_features = 7;

	LinearLayer<float, Cuda> layer( out_features );
	LatchLayer<float, Cuda> in_latch, out_latch, g_up, g_prior;
	std::size_t const n_in  = rows * in_cols;
	std::size_t const n_out = rows * out_features;

	wireCuda( layer, in_latch, out_latch, g_up, g_prior, std::vector<float>( n_in, 1.f ),
	          std::vector<float>( n_out, 0.f ), rows, in_cols, out_features );
	layer.forward();
	cudaDeviceSynchronize();

	REQUIRE( out_latch.shape()[0] == rows );
	REQUIRE( out_latch.shape()[1] == out_features );
}

TEST_CASE( "LinearLayer2 CUDA forward with known weights", "[linear2][cuda][forward][numerical]" )
{
	std::size_t const rows         = 1;
	std::size_t const in_features = 2;
	std::size_t const out_features = 2;

	std::vector<float> const w_data = { 0.5f, 0.3f, 0.2f, 0.4f };
	std::vector<float> const b_data = { 0.1f, 0.2f };
	std::vector<float> const x_data = { 1.0f, 2.0f };

	LinearLayer<float, Cuda> layer( out_features );
	LatchLayer<float, Cuda> in_latch, out_latch, g_up, g_prior;
	wireCuda( layer, in_latch, out_latch, g_up, g_prior, x_data,
	          std::vector<float>( rows * out_features, 0.f ), rows, in_features,
	          out_features );
	layer.forward();
	setCudaKnownWeights( layer, w_data, b_data );
	layer.forward();
	cudaDeviceSynchronize();

	std::vector<float> out( rows * out_features );
	cudaMemcpy( out.data(), out_latch.fwdData(), out.size() * sizeof( float ),
	            cudaMemcpyDeviceToHost );
	REQUIRE_THAT( out[0], WithinAbs( 1.0f, eps ) );
	REQUIRE_THAT( out[1], WithinAbs( 1.3f, eps ) );
}

TEST_CASE( "LinearLayer2 CUDA backward numerical", "[linear2][cuda][backward][numerical]" )
{
	std::size_t const rows         = 1;
	std::size_t const in_features = 2;
	std::size_t const out_features = 1;
	float const upstream           = -0.39f;

	std::vector<float> const w_data = { 0.6f, 0.7f };
	std::vector<float> const b_data = { 0.1f };
	std::vector<float> const x_data = { 1.0f, 1.3f };

	LinearLayer<float, Cuda> layer( out_features );
	LatchLayer<float, Cuda> in_latch, out_latch, g_up, g_prior;
	wireCuda( layer, in_latch, out_latch, g_up, g_prior, x_data, { upstream }, rows,
	          in_features, out_features );
	layer.forward();
	setCudaKnownWeights( layer, w_data, b_data );
	layer.forward();
	cudaMemcpy( g_up.fwdData(), &upstream, sizeof( float ), cudaMemcpyHostToDevice );
	layer.backward();
	cudaDeviceSynchronize();

	std::vector<float> gp( in_features );
	cudaMemcpy( gp.data(), g_prior.fwdData(), gp.size() * sizeof( float ),
	            cudaMemcpyDeviceToHost );
	REQUIRE_THAT( gp[0], WithinAbs( -0.234f, eps ) );
	REQUIRE_THAT( gp[1], WithinAbs( -0.273f, eps ) );

	std::vector<float> gw_h( in_features * out_features );
	std::vector<float> gb_h( out_features );
	cudaMemcpy( gw_h.data(), layer.getGradWeights(), gw_h.size() * sizeof( float ),
	            cudaMemcpyDeviceToHost );
	cudaMemcpy( gb_h.data(), layer.getGradBias(), gb_h.size() * sizeof( float ),
	            cudaMemcpyDeviceToHost );
	REQUIRE_THAT( gw_h[0], WithinAbs( -0.39f, eps ) );
	REQUIRE_THAT( gw_h[1], WithinAbs( -0.507f, eps ) );
	REQUIRE_THAT( gb_h[0], WithinAbs( -0.39f, eps ) );
}

#endif // NEURAL_CUDA_ENABLED
