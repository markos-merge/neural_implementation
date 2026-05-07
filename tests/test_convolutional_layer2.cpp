#include "refactor/convolutional_layer.hpp"
#include "refactor/latch_layer.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <vector>

using neural::refactor::Cpu;
using neural::refactor::LatchLayer;
using neural::refactor::ConvolutionalLayer;

TEST_CASE( "RefactorConvolutionalLayer CPU forward output shape", "[conv_refactor][cpu][forward][shape]" )
{
	std::vector<std::size_t> const in_shape{ 1u, 1u, 6u, 6u };
	std::size_t const               k      = 3;
	std::size_t const               out_ch = 2;
	std::size_t const               pad    = k / 2;
	std::size_t const               H_out  = 6u - 2 * pad;
	std::size_t const               W_out  = 6u - 2 * pad;
	std::vector<std::size_t> const  out_shape{ 1u, out_ch, H_out, W_out };

	ConvolutionalLayer<float, Cpu> layer( out_ch, k );
	LatchLayer<float, Cpu> in_l, out_l, g_dy, g_x;
	in_l.resize( in_shape );
	out_l.resize( out_shape );
	g_dy.resize( out_shape );
	g_x.resize( in_shape );

	std::vector<float> x( 1 * 1 * 6 * 6 );
	for ( std::size_t i = 0; i < x.size(); ++i )
		x[i] = std::sin( static_cast<float>( i ) * 0.13f );

	std::memcpy( in_l.fwdData(), x.data(), x.size() * sizeof( float ) );
	layer.setInputOutputLatches( &in_l, &out_l );
	layer.setGradLatches( &g_dy, &g_x );

	layer.forward();

	REQUIRE( out_l.shape().size() == 4u );
	REQUIRE( out_l.shape()[0] == 1u );
	REQUIRE( out_l.shape()[1] == out_ch );
	REQUIRE( out_l.shape()[2] == H_out );
	REQUIRE( out_l.shape()[3] == W_out );
}

TEST_CASE( "RefactorConvolutionalLayer CPU backward produces non-zero grads", "[conv_refactor][cpu][backward]" )
{
	std::vector<std::size_t> const in_shape{ 1u, 1u, 6u, 6u };
	std::size_t const               k      = 3;
	std::size_t const               out_ch = 2;
	std::size_t const               pad    = k / 2;
	std::size_t const               H_out  = 6u - 2 * pad;
	std::size_t const               W_out  = 6u - 2 * pad;
	std::vector<std::size_t> const  out_shape{ 1u, out_ch, H_out, W_out };

	ConvolutionalLayer<float, Cpu> layer( out_ch, k );
	LatchLayer<float, Cpu> in_l, out_l, g_dy, g_x;
	in_l.resize( in_shape );
	out_l.resize( out_shape );
	g_dy.resize( out_shape );
	g_x.resize( in_shape );

	std::memset(
	    in_l.fwdData(), 0, in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3] * sizeof( float ) );
	layer.setInputOutputLatches( &in_l, &out_l );
	layer.setGradLatches( &g_dy, &g_x );

	layer.forward();

	std::vector<float> upstream( out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3], 0.25f );
	std::memcpy( g_dy.fwdData(), upstream.data(), upstream.size() * sizeof( float ) );

	layer.backward();

	bool const param_grad =
	    std::abs( layer.getGradWeights()[0] ) > 0.f || std::abs( layer.getGradBias()[0] ) > 0.f;
	REQUIRE( param_grad );

	float const *gx = g_x.fwdData();
	float          s   = 0.f;
	for ( std::size_t i = 0; i < 6 * 6; ++i )
		s += std::abs( gx[i] );
	REQUIRE( s > 0.f );
}

TEST_CASE( "RefactorConvolutionalLayer ctor with input_chw pre-allocates weights", "[conv_refactor][cpu][ctor]" )
{
	ConvolutionalLayer<float, Cpu> layer( 4, 3, { 2, 8, 8 } );
	REQUIRE( layer.getWeights() != nullptr );
	REQUIRE( layer.outChannels() == 4u );
	REQUIRE( layer.kernelSize() == 3u );
}
