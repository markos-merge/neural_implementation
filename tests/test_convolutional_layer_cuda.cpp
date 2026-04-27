#include "convolutional_layer.hpp"
#include "layer_wiring_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <array>
#include <cstddef>
#include <vector>

#if NEURAL_CUDNN_ENABLED
#include "cuda_tensor_n.hpp"
#include "neural_cuda_runtime.hpp"
#include <cuda_runtime.h>
#endif

using neural::test::wire_layer;
using Catch::Matchers::WithinAbs;

#if NEURAL_CUDNN_ENABLED

namespace {

using neural::CudaTensor4;
using neural::ConvolutionalLayer;

constexpr float eps = 5e-4f;

std::array<std::size_t, 4> row_major_strides( std::array<std::size_t, 4> const &shape )
{
	std::size_t const s3 = shape[3];
	std::size_t const s2 = shape[2] * s3;
	std::size_t const s1 = shape[1] * s2;
	return { shape[1] * s2, s2, s3, 1 };
}

std::size_t flat4( std::array<std::size_t, 4> const &ix, std::array<std::size_t, 4> const &st )
{
	return ix[0] * st[0] + ix[1] * st[1] + ix[2] * st[2] + ix[3] * st[3];
}

/// Host reference matching \c CudaTensor4::im2ColConvolution (pad \f$(K-1)/2\f$, stride 1, cross-correlation).
void ref_conv_forward( std::vector<float> const &in, std::array<std::size_t, 4> const &in_shape,
                       std::vector<float> const &weights, std::array<std::size_t, 4> const &w_shape,
                       std::vector<float> const &bias_chw, std::vector<float> &out,
                       std::array<std::size_t, 4> const &out_shape )
{
	std::ptrdiff_t const pad = static_cast<std::ptrdiff_t>( w_shape[2] / 2 );
	auto const ist           = row_major_strides( in_shape );
	auto const ost           = row_major_strides( out_shape );
	auto const wst           = row_major_strides( w_shape );

	std::size_t const N  = out_shape[0];
	std::size_t const Co = out_shape[1];
	std::size_t const Oh = out_shape[2];
	std::size_t const Ow = out_shape[3];
	std::size_t const Ci = w_shape[1];
	std::size_t const Kh = w_shape[2];
	std::size_t const Kw = w_shape[3];

	for ( std::size_t n = 0; n < N; ++n ) {
		for ( std::size_t co = 0; co < Co; ++co ) {
			float const b = bias_chw[co];
			for ( std::size_t y = 0; y < Oh; ++y ) {
				for ( std::size_t x = 0; x < Ow; ++x ) {
					float acc = b;
					for ( std::size_t ci = 0; ci < Ci; ++ci ) {
						for ( std::size_t ky = 0; ky < Kh; ++ky ) {
							for ( std::size_t kx = 0; kx < Kw; ++kx ) {
								std::ptrdiff_t const iy =
								    static_cast<std::ptrdiff_t>( y ) +
								    static_cast<std::ptrdiff_t>( ky ) - pad;
								std::ptrdiff_t const ix =
								    static_cast<std::ptrdiff_t>( x ) +
								    static_cast<std::ptrdiff_t>( kx ) - pad;
								float xv = 0.f;
								if ( iy >= 0 && ix >= 0 &&
								     iy < static_cast<std::ptrdiff_t>( in_shape[2] ) &&
								     ix < static_cast<std::ptrdiff_t>( in_shape[3] ) ) {
									xv = in[flat4( { n, ci, static_cast<std::size_t>( iy ),
									                 static_cast<std::size_t>( ix ) },
									          ist )];
								}
								float const wv = weights[flat4( { co, ci, ky, kx }, wst )];
								acc += wv * xv;
							}
						}
					}
					out[flat4( { n, co, y, x }, ost )] = acc;
				}
			}
		}
	}
}

void ref_conv_backward( std::vector<float> const &grad_out, std::array<std::size_t, 4> const &go_shape,
                        std::vector<float> const &in_act, std::array<std::size_t, 4> const &in_shape,
                        std::vector<float> const &weights, std::array<std::size_t, 4> const &w_shape,
                        std::vector<float> &grad_w, std::vector<float> &grad_b,
                        std::vector<float> &grad_in )
{
	std::ptrdiff_t const pad = static_cast<std::ptrdiff_t>( w_shape[2] / 2 );
	auto const gst = row_major_strides( go_shape );
	auto const ist = row_major_strides( in_shape );
	auto const wst = row_major_strides( w_shape );

	std::fill( grad_w.begin(), grad_w.end(), 0.f );
	std::fill( grad_b.begin(), grad_b.end(), 0.f );
	std::fill( grad_in.begin(), grad_in.end(), 0.f );

	std::size_t const N  = go_shape[0];
	std::size_t const Co = go_shape[1];
	std::size_t const Oh = go_shape[2];
	std::size_t const Ow = go_shape[3];
	std::size_t const Ci = w_shape[1];
	std::size_t const Kh = w_shape[2];
	std::size_t const Kw = w_shape[3];

	for ( std::size_t n = 0; n < N; ++n ) {
		for ( std::size_t co = 0; co < Co; ++co ) {
			for ( std::size_t y = 0; y < Oh; ++y ) {
				for ( std::size_t x = 0; x < Ow; ++x ) {
					float const dy = grad_out[flat4( { n, co, y, x }, gst )];
					grad_b[co] += dy;
					for ( std::size_t ci = 0; ci < Ci; ++ci ) {
						for ( std::size_t ky = 0; ky < Kh; ++ky ) {
							for ( std::size_t kx = 0; kx < Kw; ++kx ) {
								std::ptrdiff_t const iy =
								    static_cast<std::ptrdiff_t>( y ) +
								    static_cast<std::ptrdiff_t>( ky ) - pad;
								std::ptrdiff_t const ix =
								    static_cast<std::ptrdiff_t>( x ) +
								    static_cast<std::ptrdiff_t>( kx ) - pad;
								float xv = 0.f;
								if ( iy >= 0 && ix >= 0 &&
								     iy < static_cast<std::ptrdiff_t>( in_shape[2] ) &&
								     ix < static_cast<std::ptrdiff_t>( in_shape[3] ) ) {
									xv = in_act[flat4( { n, ci, static_cast<std::size_t>( iy ),
									                     static_cast<std::size_t>( ix ) },
									              ist )];
									grad_in[flat4( { n, ci, static_cast<std::size_t>( iy ),
									                 static_cast<std::size_t>( ix ) },
									          ist )] += dy * weights[flat4( { co, ci, ky, kx }, wst )];
								}
								grad_w[flat4( { co, ci, ky, kx }, wst )] += dy * xv;
							}
						}
					}
				}
			}
		}
	}
}

std::vector<float> tensor4_to_host( CudaTensor4<float> const &t )
{
	std::vector<float> v( t.size() );
	if ( !v.empty() &&
	     cudaMemcpy( v.data(), t.data(), v.size() * sizeof( float ), cudaMemcpyDeviceToHost ) !=
	         cudaSuccess ) {
		throw std::runtime_error( "tensor4_to_host: cudaMemcpy failed" );
	}
	return v;
}

float read4( CudaTensor4<float> const &t, std::size_t n, std::size_t c, std::size_t h, std::size_t w )
{
	return t( std::array<std::size_t, 4>{ n, c, h, w } );
}

} // namespace

TEST_CASE( "ConvolutionalLayer CudaTensor4 forward output shape", "[conv_layer][cuda4][shape]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::array<std::size_t, 4> const in_shape = { 2u, 3u, 8u, 8u };
	std::size_t const out_ch                 = 4u;
	std::size_t const k                      = 3u;
	std::array<std::size_t, 4> const out_shape = { in_shape[0], out_ch, in_shape[2] + 3u - k,
		                                              in_shape[3] + 3u - k };

	ConvolutionalLayer<CudaTensor4<float>> layer( out_ch, k );
	CudaTensor4<float> in_buf( in_shape, 0.1f );
	CudaTensor4<float> out_buf( out_shape, 0.f );
	CudaTensor4<float> g_out( out_shape, 0.f );
	CudaTensor4<float> g_in( in_shape, 0.f );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );

	layer.forward();
	REQUIRE( layer.getWeights().shape()[0] == out_ch );
	REQUIRE( layer.getWeights().shape()[1] == in_shape[1] );
	REQUIRE( layer.getWeights().shape()[2] == k );
	REQUIRE( layer.getOutput()->shape() == out_shape );
}

TEST_CASE( "ConvolutionalLayer CudaTensor4 forward matches host reference", "[conv_layer][cuda4][forward]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::array<std::size_t, 4> const in_shape = { 1u, 2u, 5u, 5u };
	std::size_t const out_ch                 = 3u;
	std::size_t const k                      = 3u;
	std::array<std::size_t, 4> const out_shape = { in_shape[0], out_ch, in_shape[2] + 3u - k,
		                                              in_shape[3] + 3u - k };

	std::vector<float> in_h( in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3] );
	for ( std::size_t i = 0; i < in_h.size(); ++i ) {
		in_h[i] = 0.02f * static_cast<float>( static_cast<int>( i % 17 ) - 8 );
	}
	std::array<std::size_t, 4> const w_shape = { out_ch, in_shape[1], k, k };
	std::vector<float> w_h( out_ch * in_shape[1] * k * k );
	for ( std::size_t i = 0; i < w_h.size(); ++i ) {
		w_h[i] = 0.01f * static_cast<float>( static_cast<int>( i % 11 ) - 5 );
	}
	std::vector<float> b_h( out_ch );
	for ( std::size_t c = 0; c < out_ch; ++c ) {
		b_h[c] = 0.03f * static_cast<float>( static_cast<int>( c ) - 1 );
	}

	std::vector<float> out_ref( out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] );
	ref_conv_forward( in_h, in_shape, w_h, w_shape, b_h, out_ref, out_shape );

	ConvolutionalLayer<CudaTensor4<float>> layer( out_ch, k );
	CudaTensor4<float> in_buf( in_shape );
	in_buf.assign( in_h );
	CudaTensor4<float> out_buf( out_shape, 0.f );
	CudaTensor4<float> g_out( out_shape, 0.f );
	CudaTensor4<float> g_in( in_shape, 0.f );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );

	layer.forward();
	layer.getWeights().assign( w_h );
	std::vector<float> bias_rank4( out_ch );
	for ( std::size_t c = 0; c < out_ch; ++c ) {
		bias_rank4[c] = b_h[c];
	}
	layer.getBias().assign( bias_rank4 );
	layer.forward();

	std::vector<float> const out_gpu = tensor4_to_host( out_buf );
	REQUIRE( out_gpu.size() == out_ref.size() );
	for ( std::size_t i = 0; i < out_ref.size(); ++i ) {
		REQUIRE_THAT( out_gpu[i], WithinAbs( out_ref[i], eps ) );
	}
}

TEST_CASE( "ConvolutionalLayer CudaTensor4 backward matches host reference", "[conv_layer][cuda4][backward]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::array<std::size_t, 4> const in_shape = { 1u, 2u, 4u, 4u };
	std::size_t const out_ch                 = 2u;
	std::size_t const k                      = 3u;
	std::array<std::size_t, 4> const out_shape = { in_shape[0], out_ch, in_shape[2] + 3u - k,
		                                              in_shape[3] + 3u - k };

	std::vector<float> in_h( in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3] );
	for ( std::size_t i = 0; i < in_h.size(); ++i ) {
		in_h[i] = 0.07f * static_cast<float>( i ) - 0.5f;
	}
	std::array<std::size_t, 4> const w_shape = { out_ch, in_shape[1], k, k };
	std::vector<float> w_h( w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3] );
	for ( std::size_t i = 0; i < w_h.size(); ++i ) {
		w_h[i] = 0.05f * static_cast<float>( ( i + 3 ) % 7 ) - 0.1f;
	}
	std::vector<float> b_h( out_ch, 0.f );

	std::vector<float> grad_out_h( out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] );
	for ( std::size_t i = 0; i < grad_out_h.size(); ++i ) {
		grad_out_h[i] = ( i % 2 == 0 ) ? 1.f : -0.5f;
	}

	std::vector<float> gw_ref( w_h.size(), 0.f );
	std::vector<float> gb_ref( out_ch, 0.f );
	std::vector<float> gx_ref( in_h.size(), 0.f );
	ref_conv_backward( grad_out_h, out_shape, in_h, in_shape, w_h, w_shape, gw_ref, gb_ref, gx_ref );

	ConvolutionalLayer<CudaTensor4<float>> layer( out_ch, k );
	CudaTensor4<float> in_buf( in_shape );
	in_buf.assign( in_h );
	CudaTensor4<float> out_buf( out_shape, 0.f );
	CudaTensor4<float> g_out( out_shape );
	g_out.assign( grad_out_h );
	CudaTensor4<float> g_in( in_shape, 0.f );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );

	layer.forward();
	layer.getWeights().assign( w_h );
	layer.getBias().assign( b_h );
	layer.forward();
	layer.backward();

	std::vector<float> const gw_gpu = tensor4_to_host( layer.getGradWeights() );
	std::vector<float> const gx_gpu = tensor4_to_host( g_in );
	std::vector<float> const gb_tensor = tensor4_to_host( layer.getGradBias() );

	REQUIRE( gw_gpu.size() == gw_ref.size() );
	for ( std::size_t i = 0; i < gw_ref.size(); ++i ) {
		REQUIRE_THAT( gw_gpu[i], WithinAbs( gw_ref[i], eps ) );
	}
	REQUIRE( gx_gpu.size() == gx_ref.size() );
	for ( std::size_t i = 0; i < gx_ref.size(); ++i ) {
		REQUIRE_THAT( gx_gpu[i], WithinAbs( gx_ref[i], eps ) );
	}
	for ( std::size_t c = 0; c < out_ch; ++c ) {
		REQUIRE_THAT( gb_tensor[c], WithinAbs( gb_ref[c], eps ) );
	}
}

TEST_CASE( "ConvolutionalLayer CudaTensor4 bias added on forward", "[conv_layer][cuda4][bias]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::array<std::size_t, 4> const in_shape = { 1u, 1u, 3u, 3u };
	std::size_t const out_ch                 = 1u;
	std::size_t const k                      = 3u;
	std::array<std::size_t, 4> const out_shape = { 1u, 1u, 3u, 3u };

	ConvolutionalLayer<CudaTensor4<float>> layer( out_ch, k );
	CudaTensor4<float> in_buf( in_shape, 0.f );
	CudaTensor4<float> out_buf( out_shape, 0.f );
	CudaTensor4<float> g_out( out_shape, 0.f );
	CudaTensor4<float> g_in( in_shape, 0.f );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	std::vector<float> w( k * k, 0.f );
	w[k * k / 2] = 1.f;
	layer.getWeights().assign( w );
	std::vector<float> b = { 2.5f };
	layer.getBias().assign( b );
	in_buf.assign( std::vector<float>( in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3], 0.f ) );
	layer.forward();

	REQUIRE_THAT( read4( out_buf, 0, 0, 1, 1 ), WithinAbs( 2.5f, eps ) );
}

#endif // NEURAL_CUDNN_ENABLED
