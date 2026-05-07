#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <array>
#include <cstddef>
#include <numeric>
#include <vector>
#include <stdexcept>

#if NEURAL_CUDNN_ENABLED
#include "cuda_tensor_n.hpp"
#include "neural_cuda_runtime.hpp"
#include <cuda_runtime.h>
#endif

using Catch::Matchers::WithinAbs;

#if NEURAL_CUDNN_ENABLED

namespace {

using neural::CudaTensor4;
using neural::CudaTensorNBase;

constexpr float eps = 1e-4f;

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

float read1( CudaTensorNBase<1, float> const &t, std::size_t i )
{
	return t( std::array<std::size_t, 1>{ i } );
}

/// Same pad \f$(K-1)/2\f$, stride 1 as \c CudaTensor4::im2ColConvolution (cross-correlation).
float ref_conv_out( std::vector<float> const &in, std::array<std::size_t, 4> const &in_shape,
                    std::vector<float> const &ker, std::size_t Kh, std::size_t Kw, std::size_t n,
                    std::size_t co, std::size_t y, std::size_t x, std::ptrdiff_t pad )
{
	std::array<std::size_t, 4> const ist = { in_shape[1] * in_shape[2] * in_shape[3],
		                                        in_shape[2] * in_shape[3], in_shape[3], 1u };
	std::array<std::size_t, 4> const w_shape = { 1u, in_shape[1], Kh, Kw };
	std::array<std::size_t, 4> const wst = { w_shape[1] * Kh * Kw, Kh * Kw, Kw, 1u };
	float acc = 0.f;
	std::size_t const Ci = in_shape[1];
	for ( std::size_t ci = 0; ci < Ci; ++ci ) {
		for ( std::size_t ky = 0; ky < Kh; ++ky ) {
			for ( std::size_t kx = 0; kx < Kw; ++kx ) {
				std::ptrdiff_t const iy =
				    static_cast<std::ptrdiff_t>( y ) + static_cast<std::ptrdiff_t>( ky ) - pad;
				std::ptrdiff_t const ix =
				    static_cast<std::ptrdiff_t>( x ) + static_cast<std::ptrdiff_t>( kx ) - pad;
				float xv = 0.f;
				if ( iy >= 0 && ix >= 0 && iy < static_cast<std::ptrdiff_t>( in_shape[2] ) &&
				     ix < static_cast<std::ptrdiff_t>( in_shape[3] ) ) {
					std::size_t const fi = n * ist[0] + ci * ist[1] +
					                       static_cast<std::size_t>( iy ) * ist[2] +
					                       static_cast<std::size_t>( ix ) * ist[3];
					xv = in[fi];
				}
				std::size_t const wf = co * wst[0] + ci * wst[1] + ky * wst[2] + kx * wst[3];
				acc += ker[wf] * xv;
			}
		}
	}
	return acc;
}

} // namespace

TEST_CASE( "CudaTensor4 default is empty", "[cuda_tensor4][shape]" )
{
	CudaTensor4<float> t;
	REQUIRE( t.size() == 0u );
}

TEST_CASE( "CudaTensor4 shape size and row-major strides", "[cuda_tensor4][shape]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::array<std::size_t, 4> const sh = { 2u, 3u, 4u, 5u };
	CudaTensor4<float> t( sh, 0.f );
	REQUIRE( t.shape() == sh );
	REQUIRE( t.size() == 2u * 3u * 4u * 5u );
	auto const st = t.strides();
	REQUIRE( st[3] == 1u );
	REQUIRE( st[2] == 5u );
	REQUIRE( st[1] == 4u * 5u );
	REQUIRE( st[0] == 3u * 4u * 5u );
}

TEST_CASE( "CudaTensor4 assign and read round-trip", "[cuda_tensor4][assign]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::array<std::size_t, 4> const sh = { 1u, 1u, 2u, 2u };
	std::vector<float> host( 4u );
	for ( std::size_t i = 0; i < 4u; ++i ) {
		host[i] = 0.25f * static_cast<float>( i + 1 );
	}
	CudaTensor4<float> t( sh );
	t.assign( host );
	for ( std::size_t i = 0; i < 4u; ++i ) {
		std::size_t const w = i % 2u;
		std::size_t const h = ( i / 2u ) % 2u;
		REQUIRE_THAT( read4( t, 0, 0, h, w ), WithinAbs( host[i], eps ) );
	}
}

TEST_CASE( "CudaTensor4 reshape preserves linear layout", "[cuda_tensor4][reshape]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::array<std::size_t, 4> const sh = { 1u, 2u, 1u, 3u };
	std::vector<float> host( 6u );
	std::iota( host.begin(), host.end(), 0.f );
	CudaTensor4<float> t( sh );
	t.assign( host );
	t.reshape( { 1u, 1u, 2u, 3u } );
	REQUIRE( t.shape() == ( std::array<std::size_t, 4>{ 1u, 1u, 2u, 3u } ) );
	std::vector<float> const back = tensor4_to_host( t );
	REQUIRE( back == host );
}

TEST_CASE( "CudaTensor4 copy constructor copies values", "[cuda_tensor4][copy]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	CudaTensor4<float> a( std::array<std::size_t, 4>{ 1u, 1u, 1u, 1u }, 3.25f );
	CudaTensor4<float> b( a );
	REQUIRE_THAT( read4( b, 0, 0, 0, 0 ), WithinAbs( 3.25f, eps ) );
}

TEST_CASE( "CudaTensor4 elementwiseMultiply into output tensor", "[cuda_tensor4][mul]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::array<std::size_t, 4> const sh = { 1u, 1u, 2u, 2u };
	CudaTensor4<float> a( sh, 0.f );
	a.assign( std::array<std::size_t, 4>{ 0, 0, 0, 0 }, 2.f );
	a.assign( std::array<std::size_t, 4>{ 0, 0, 1, 1 }, 3.f );
	CudaTensor4<float> b( sh, 0.f );
	b.assign( std::array<std::size_t, 4>{ 0, 0, 0, 0 }, 4.f );
	b.assign( std::array<std::size_t, 4>{ 0, 0, 1, 1 }, 0.5f );
	CudaTensor4<float> out( sh, 999.f );
	a.elementwiseMultiply( b, out );
	REQUIRE_THAT( read4( out, 0, 0, 0, 0 ), WithinAbs( 8.f, eps ) );
	REQUIRE_THAT( read4( out, 0, 0, 1, 1 ), WithinAbs( 1.5f, eps ) );
}

TEST_CASE( "CudaTensor4 reduceSumToDim axis 1", "[cuda_tensor4][reduce]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::array<std::size_t, 4> const sh = { 2u, 3u, 1u, 1u };
	CudaTensor4<float> t( sh, 1.f );
	CudaTensorNBase<1, float> sum1;
	t.reduceSumToDim( 1u, sum1 );
	REQUIRE( sum1.shape()[0] == 3u );
	for ( std::size_t c = 0; c < 3u; ++c ) {
		REQUIRE_THAT( read1( sum1, c ), WithinAbs( 2.f, eps ) );
	}
}

TEST_CASE( "CudaTensor4 addColorChannelInPlace rank-1 bias", "[cuda_tensor4][bias]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::array<std::size_t, 4> const sh = { 1u, 2u, 2u, 2u };
	CudaTensor4<float> t( sh, 1.f );
	std::vector<float> const bdata = { 0.5f, 3.f };
	CudaTensorNBase<1, float> bias( { 2u } );
	bias.assign( bdata );
	t.addColorChannelInPlace( bias );
	for ( std::size_t h = 0; h < 2u; ++h ) {
		for ( std::size_t w = 0; w < 2u; ++w ) {
			REQUIRE_THAT( read4( t, 0, 0, h, w ), WithinAbs( 1.5f, eps ) );
			REQUIRE_THAT( read4( t, 0, 1, h, w ), WithinAbs( 4.f, eps ) );
		}
	}
}

TEST_CASE( "CudaTensor4 addColorChannelInPlace rank-4 bias", "[cuda_tensor4][bias]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::array<std::size_t, 4> const sh = { 1u, 2u, 1u, 1u };
	CudaTensor4<float> t( sh, 2.f );
	CudaTensor4<float> b( std::array<std::size_t, 4>{ 1u, 2u, 1u, 1u } );
	b.assign( std::array<std::size_t, 4>{ 0, 0, 0, 0 }, 1.f );
	b.assign( std::array<std::size_t, 4>{ 0, 1, 0, 0 }, 10.f );
	t.addColorChannelInPlace( b );
	REQUIRE_THAT( read4( t, 0, 0, 0, 0 ), WithinAbs( 3.f, eps ) );
	REQUIRE_THAT( read4( t, 0, 1, 0, 0 ), WithinAbs( 12.f, eps ) );
}

TEST_CASE( "CudaTensor4 operator*= elementwise", "[cuda_tensor4][mul_inplace]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::array<std::size_t, 4> const sh = { 1u, 1u, 2u, 2u };
	CudaTensor4<float> a( sh, 2.f );
	CudaTensor4<float> b( sh, 0.f );
	b.assign( std::array<std::size_t, 4>{ 0, 0, 0, 0 }, 0.5f );
	b.assign( std::array<std::size_t, 4>{ 0, 0, 0, 1 }, 3.f );
	b.assign( std::array<std::size_t, 4>{ 0, 0, 1, 0 }, -1.f );
	b.assign( std::array<std::size_t, 4>{ 0, 0, 1, 1 }, 4.f );
	a *= b;
	REQUIRE_THAT( read4( a, 0, 0, 0, 0 ), WithinAbs( 1.f, eps ) );
	REQUIRE_THAT( read4( a, 0, 0, 0, 1 ), WithinAbs( 6.f, eps ) );
	REQUIRE_THAT( read4( a, 0, 0, 1, 0 ), WithinAbs( -2.f, eps ) );
	REQUIRE_THAT( read4( a, 0, 0, 1, 1 ), WithinAbs( 8.f, eps ) );
}

TEST_CASE( "CudaTensor4 operator*= scalar", "[cuda_tensor4][scale]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	CudaTensor4<float> t( std::array<std::size_t, 4>{ 1u, 1u, 1u, 2u }, -3.f );
	t *= 0.5f;
	REQUIRE_THAT( read4( t, 0, 0, 0, 0 ), WithinAbs( -1.5f, eps ) );
	REQUIRE_THAT( read4( t, 0, 0, 0, 1 ), WithinAbs( -1.5f, eps ) );
}

TEST_CASE( "CudaTensor4 im2ColConvolution matches host reference (spot checks)", "[cuda_tensor4][conv]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::array<std::size_t, 4> const in_shape = { 1u, 1u, 4u, 4u };
	std::size_t const k = 3u;
	std::ptrdiff_t const pad = static_cast<std::ptrdiff_t>( k / 2 );
	std::vector<float> in_h( in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3], 1.f );
	std::vector<float> w_h( k * k, 1.f );

	CudaTensor4<float> input( in_shape );
	input.assign( in_h );
	CudaTensor4<float> kernel( std::array<std::size_t, 4>{ 1u, 1u, k, k } );
	kernel.assign( w_h );
	CudaTensor4<float> output;
	CudaTensorNBase<2, float> im2col;
	CudaTensor4<float> gemm;
	input.im2ColConvolution( kernel, im2col, output, gemm );

	REQUIRE( output.shape()[2] == in_shape[2] );
	REQUIRE( output.shape()[3] == in_shape[3] );

	struct {
		std::size_t y, x;
		float expected;
	} spots[] = { { 0, 0, 4.f }, { 1, 1, 9.f }, { 3, 3, 4.f } };
	for ( auto const &s : spots ) {
		float const ref =
		    ref_conv_out( in_h, in_shape, w_h, k, k, 0, 0, s.y, s.x, pad );
		REQUIRE_THAT( ref, WithinAbs( s.expected, eps ) );
		REQUIRE_THAT( read4( output, 0, 0, s.y, s.x ), WithinAbs( s.expected, eps ) );
	}
}

TEST_CASE( "CudaTensor4 pointer ctor own=false aliases device storage", "[cuda_tensor4][view]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::array<std::size_t, 4> const sh = { 1u, 1u, 2u, 2u };
	CudaTensor4<float> src( sh, 1.f );
	float *const raw = src.data();
	CudaTensor4<float> view( raw, sh, false );
	std::vector<float> const upd( 4u, 7.f );
	src.assign( upd );
	std::vector<float> const w = tensor4_to_host( view );
	REQUIRE_THAT( w[0], WithinAbs( 7.f, eps ) );
}

TEST_CASE( "CudaTensor4 pointer ctor own=true copies device storage", "[cuda_tensor4][own]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::array<std::size_t, 4> const sh = { 1u, 1u, 2u, 2u };
	CudaTensor4<float> src( sh, 2.f );
	float *const raw = src.data();
	CudaTensor4<float> dup( raw, sh, true );
	(void)cudaMemset( raw, 0, 4u * sizeof( float ) );
	std::vector<float> const w = tensor4_to_host( dup );
	REQUIRE_THAT( w[0], WithinAbs( 2.f, eps ) );
}

TEST_CASE( "CudaTensor4 non-owning copy-assign requires matching shape", "[cuda_tensor4][view]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	float *d = nullptr;
	REQUIRE( cudaMalloc( reinterpret_cast<void **>( &d ), 4u * sizeof( float ) ) == cudaSuccess );
	std::array<std::size_t, 4> const sh = { 1u, 1u, 2u, 2u };
	CudaTensor4<float> v( d, sh, false );
	CudaTensor4<float> rhs( std::array<std::size_t, 4>{ 1u, 1u, 1u, 4u }, 1.f );
	REQUIRE_THROWS_AS( v = rhs, std::invalid_argument );
	(void)cudaFree( d );
}

TEST_CASE( "CudaTensor4 non-owning move-assign throws", "[cuda_tensor4][view]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	float *d = nullptr;
	REQUIRE( cudaMalloc( reinterpret_cast<void **>( &d ), 4u * sizeof( float ) ) == cudaSuccess );
	std::array<std::size_t, 4> const sh = { 1u, 1u, 2u, 2u };
	CudaTensor4<float> v( d, sh, false );
	CudaTensor4<float> rhs( sh, 1.f );
	REQUIRE_THROWS_AS( v = std::move( rhs ), std::invalid_argument );
	(void)cudaFree( d );
}

TEST_CASE( "CudaTensor4 elementwiseMultiply throws if non-owning out would resize", "[cuda_tensor4][view]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::array<std::size_t, 4> const sh = { 1u, 1u, 2u, 2u };
	CudaTensor4<float> a( sh, 2.f );
	CudaTensor4<float> b( sh, 3.f );
	float *d = nullptr;
	REQUIRE( cudaMalloc( reinterpret_cast<void **>( &d ), sizeof( float ) ) == cudaSuccess );
	CudaTensor4<float> out( d, std::array<std::size_t, 4>{ 1u, 1u, 1u, 1u }, false );
	REQUIRE_THROWS_AS( a.elementwiseMultiply( b, out ), std::invalid_argument );
	(void)cudaFree( d );
}

#endif // NEURAL_CUDNN_ENABLED
