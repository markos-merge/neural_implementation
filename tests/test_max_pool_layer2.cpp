#include "refactor/latch_layer.hpp"
#include "refactor/max_pool_layer.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstring>
#include <vector>

#if NEURAL_CUDNN_ENABLED
#include <cuda_runtime.h>
#endif

using neural::refactor::Cpu;
using neural::refactor::LatchLayer;
using neural::refactor::MaxPoolLayer;
using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps = 1e-5f;

static std::vector<std::size_t> shape4( std::size_t n, std::size_t c, std::size_t h, std::size_t w )
{
	return { n, c, h, w };
}

// 1×1×4×4 NCHW, forward output 1×1×2×2 for pool=stride=2
void wire_cpu_4(
    MaxPoolLayer<float, Cpu> &layer,
    LatchLayer<float, Cpu> &  in_l,
    LatchLayer<float, Cpu> &  out_l,
    LatchLayer<float, Cpu> &  g_up_l,
    LatchLayer<float, Cpu> &  g_prior_l,
    std::vector<float> const &input_data )
{
	auto const si         = shape4( 1, 1, 4, 4 );
	auto const forward_out = shape4( 1, 1, 2, 2 );

	in_l.resize( si );
	out_l.resize( forward_out );
	g_up_l.resize( forward_out );
	g_prior_l.resize( si );

	std::memcpy( in_l.fwdData(), input_data.data(), input_data.size() * sizeof( float ) );

	layer.setInputOutputLatches( &in_l, &out_l );
	layer.setGradLatches( &g_up_l, &g_prior_l );
}

#if NEURAL_CUDNN_ENABLED
using neural::refactor::Cuda;

static void wire_cuda_4(
    MaxPoolLayer<float, Cuda> &layer,
    LatchLayer<float, Cuda> &  in_l,
    LatchLayer<float, Cuda> &  out_l,
    LatchLayer<float, Cuda> &  g_up_l,
    LatchLayer<float, Cuda> &  g_prior_l,
    std::vector<float> const &input_data )
{
	auto const si          = shape4( 1, 1, 4, 4 );
	auto const forward_out = shape4( 1, 1, 2, 2 );

	in_l.resize( si );
	out_l.resize( forward_out );
	g_up_l.resize( forward_out );
	g_prior_l.resize( si );

	cudaMemcpy( in_l.fwdData(), input_data.data(), input_data.size() * sizeof( float ),
	            cudaMemcpyHostToDevice );

	layer.setInputOutputLatches( &in_l, &out_l );
	layer.setGradLatches( &g_up_l, &g_prior_l );
}

#endif // NEURAL_CUDNN_ENABLED

} // namespace

// Input (single channel):  1  2  3  4 / 5  6  7  8 / 9 10 11 12 / 13 14 15 16
// 2x2 pool stride 2 -- max outputs: 6, 8, 14, 16

TEST_CASE( "MaxPoolLayer2 CPU forward outputs expected maxima", "[max_pool2][cpu][forward]" )
{
	MaxPoolLayer<float, Cpu> layer( 2, 2 );

	std::vector<float> const hi = {
	    1.f,  2.f,  3.f,  4.f,  //
	    5.f,  6.f,  7.f,  8.f,  //
	    9.f,  10.f, 11.f, 12.f, //
	    13.f, 14.f, 15.f, 16.f  //
	};

	LatchLayer<float, Cpu> in_l, out_l, g_up, g_prior;
	wire_cpu_4( layer, in_l, out_l, g_up, g_prior, hi );
	layer.forward();

	float const *yo = out_l.fwdData();
	REQUIRE_THAT( yo[0], WithinAbs( 6.f, eps ) );
	REQUIRE_THAT( yo[1], WithinAbs( 8.f, eps ) );
	REQUIRE_THAT( yo[2], WithinAbs( 14.f, eps ) );
	REQUIRE_THAT( yo[3], WithinAbs( 16.f, eps ) );

	REQUIRE( layer.poolSize() == 2u );
	REQUIRE( layer.stride() == 2u );
}

TEST_CASE( "MaxPoolLayer2 CPU backward routes gradient to argmax winners", "[max_pool2][cpu][backward]" )
{
	MaxPoolLayer<float, Cpu> layer( 2, 2 );

	std::vector<float> const hi = {
	    1.f,  2.f,  3.f,  4.f,   //
	    5.f,  6.f,  7.f,  8.f,   //
	    9.f,  10.f, 11.f, 12.f,  //
	    13.f, 14.f, 15.f, 16.f   //
	};

	LatchLayer<float, Cpu> in_l, out_l, g_up, g_prior;
	wire_cpu_4( layer, in_l, out_l, g_up, g_prior, hi );
	layer.forward();

	std::vector<float> upstream = { -1.f, 1.f, 2.f, -2.f }; // [ (0,0), (0,1), (1,0), (1,1) ]
	std::memcpy( g_up.fwdData(), upstream.data(), upstream.size() * sizeof( float ) );
	layer.backward();

	// Winning flat indices within the 4×4 map: (1,1)=6, (1,3)=8, (3,1)=14, (3,3)=16
	float const *gx = g_prior.fwdData();
	REQUIRE_THAT( gx[5], WithinAbs( -1.f, eps ) );
	REQUIRE_THAT( gx[7], WithinAbs( 1.f, eps ) );
	REQUIRE_THAT( gx[13], WithinAbs( 2.f, eps ) );
	REQUIRE_THAT( gx[15], WithinAbs( -2.f, eps ) );

	std::vector<float> expect( 16, 0.f );
	expect[5]  = -1.f;
	expect[7]  = 1.f;
	expect[13] = 2.f;
	expect[15] = -2.f;
	for ( std::size_t i = 0; i < 16; ++i )
		REQUIRE_THAT( gx[i], WithinAbs( expect[i], eps ) );
}

#if NEURAL_CUDNN_ENABLED

TEST_CASE( "MaxPoolLayer2 CUDA forward outputs expected maxima", "[max_pool2][cuda][forward]" )
{
	MaxPoolLayer<float, Cuda> layer( 2, 2 );

	std::vector<float> const hi = {
	    1.f,  2.f,  3.f,  4.f,   //
	    5.f,  6.f,  7.f,  8.f,   //
	    9.f,  10.f, 11.f, 12.f,  //
	    13.f, 14.f, 15.f, 16.f   //
	};

	LatchLayer<float, Cuda> in_l, out_l, g_up, g_prior;
	wire_cuda_4( layer, in_l, out_l, g_up, g_prior, hi );
	layer.forward();
	cudaDeviceSynchronize();

	std::vector<float> yo( 4 );
	cudaMemcpy( yo.data(), out_l.fwdData(), 4 * sizeof( float ), cudaMemcpyDeviceToHost );
	REQUIRE_THAT( yo[0], WithinAbs( 6.f, eps ) );
	REQUIRE_THAT( yo[1], WithinAbs( 8.f, eps ) );
	REQUIRE_THAT( yo[2], WithinAbs( 14.f, eps ) );
	REQUIRE_THAT( yo[3], WithinAbs( 16.f, eps ) );
}

TEST_CASE( "MaxPoolLayer2 CUDA backward matches routed CPU reference", "[max_pool2][cuda][backward]" )
{
	MaxPoolLayer<float, Cuda> layer( 2, 2 );

	std::vector<float> const hi = {
	    1.f,  2.f,  3.f,  4.f,   //
	    5.f,  6.f,  7.f,  8.f,   //
	    9.f,  10.f, 11.f, 12.f,  //
	    13.f, 14.f, 15.f, 16.f   //
	};

	LatchLayer<float, Cuda> in_l, out_l, g_up, g_prior;
	wire_cuda_4( layer, in_l, out_l, g_up, g_prior, hi );
	layer.forward();

	std::vector<float> upstream = { -1.f, 1.f, 2.f, -2.f };
	cudaMemcpy( g_up.fwdData(), upstream.data(), upstream.size() * sizeof( float ),
	            cudaMemcpyHostToDevice );

	layer.backward();
	cudaDeviceSynchronize();

	std::vector<float> gx( 16 );
	cudaMemcpy( gx.data(), g_prior.fwdData(), gx.size() * sizeof( float ), cudaMemcpyDeviceToHost );

	std::vector<float> expect( 16, 0.f );
	expect[5]  = -1.f;
	expect[7]  = 1.f;
	expect[13] = 2.f;
	expect[15] = -2.f;
	for ( std::size_t i = 0; i < 16; ++i )
		REQUIRE_THAT( gx[i], WithinAbs( expect[i], eps ) );
}

#endif // NEURAL_CUDNN_ENABLED
