#include "cuda_tensor.hpp"
#include "neural_cuda_runtime.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <random>
#include <utility>
#include <vector>

using neural::CudaTensor;
using Catch::Matchers::WithinAbs;

TEST_CASE( "CudaTensor default is empty", "[cuda_tensor][shape]" )
{
	CudaTensor<float> t;
	REQUIRE( t.rows() == 0u );
	REQUIRE( t.cols() == 0u );
	REQUIRE( t.size() == 0u );
}

TEST_CASE( "CudaTensor sized construction records shape", "[cuda_tensor][shape]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	CudaTensor<float> t( 3u, 7u );
	REQUIRE( t.rows() == 3u );
	REQUIRE( t.cols() == 7u );
	REQUIRE( t.size() == 21u );
}

TEST_CASE( "CudaTensor construction from iterators records shape", "[cuda_tensor][shape]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::vector<float> data( 12, 1.0f );
	CudaTensor<float> t( 3u, 4u, data.begin(), data.end() );
	REQUIRE( t.rows() == 3u );
	REQUIRE( t.cols() == 4u );
	REQUIRE( t.size() == 12u );
}

TEST_CASE( "CudaTensor constant fill construction records shape", "[cuda_tensor][shape]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	CudaTensor<float> t( 2u, 5u, 3.14f );
	REQUIRE( t.rows() == 2u );
	REQUIRE( t.cols() == 5u );
	REQUIRE( t.size() == 10u );
}

TEST_CASE( "CudaTensor element read/write via cudaMemcpy", "[cuda_tensor][access]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	CudaTensor<float> t( 2u, 3u );
	REQUIRE( t( 0, 0 ) == 0.f );
	REQUIRE( std::as_const( t )( 1, 2 ) == 0.f );
	t.assign( 0, 0, 9.f );
	REQUIRE_THAT( t( 0, 0 ), WithinAbs( 9.f, 1e-5f ) );
}

TEST_CASE( "CudaTensor stub operations are callable", "[cuda_tensor][stub]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	{
		CudaTensor<float> a( 2u, 3u, 0.f );
		CudaTensor<float> b( 3u, 2u, 0.f );
		(void)a.matmul( b );
	}
	{
		CudaTensor<float> a( 2u, 3u, 0.f );
		CudaTensor<float> b( 3u, 2u, 0.f );
		(void)a.matmulInPlace( b );
	}
	{
		CudaTensor<float> a( 2u, 3u, 0.f );
		(void)a.transpose();
		(void)a.transposeInPlace();
		(void)a.reshape( 1u, 6u );
		(void)a.reshapeInPlace( 1u, 6u );
	}
	{
		CudaTensor<float> a( 2u, 3u, 0.f );
		CudaTensor<float> div( 2u, 1u, 1.f );
		(void)a.divideRowsWithColInPlace( div );
	}
	{
		CudaTensor<float> a( 2u, 3u, 0.f );
		(void)a.maxCoeff();
	}
	CudaTensor<float> a( 2u, 3u, 0.f );
	CudaTensor<float> b( 2u, 3u, 0.f );
	{
		CudaTensor<float> src( 2u, 3u, 0.f );
		CudaTensor<float> dst( 2u, 3u, 0.f );
		std::vector<int> const idx{ 0 };
		dst.assignTensorBlock( src, idx, 0u, 1u );
	}
	CudaTensor<float> col_bias( 3u, 1u, 0.f );
	CudaTensor<float> row_sub( 2u, 1u, 0.f );
	(void)( a + b );
	(void)( a += b );
	(void)( a - b );
	(void)( a -= b );
	(void)a.addColwise( col_bias );
	(void)a.addColwiseInPlace( col_bias );
	(void)a.subtractColwise( row_sub );
	(void)( a * b );
	(void)( a *= b );
	(void)( a * 2.f );
	(void)( a *= 2.f );
	(void)a.cwiseGreater( 0.f );
	{
		CudaTensor<float> g( 2u, 3u, 0.f );
		(void)g.cwiseGreaterInPlace( a, 0.f );
	}
	(void)a.cwiseOneMinus();
	(void)a.cwiseSigmoid();
	(void)a.cwiseExp();
	(void)a.cwiseLog();
	(void)a.sum();
	(void)a.sumAlongAxis( 0u );
	{
		CudaTensor<float> out_axis0( 1u, 3u, 0.f );
		(void)out_axis0.sumAlongAxisInPlace( a, 0u );
		CudaTensor<float> out_axis0t( 3u, 1u, 0.f );
		(void)out_axis0t.sumAlongAxisInPlace( a, 0u, true );
		CudaTensor<float> out_axis1( 2u, 1u, 0.f );
		(void)out_axis1.sumAlongAxisInPlace( a, 1u );
		CudaTensor<float> out_axis1t( 1u, 2u, 0.f );
		(void)out_axis1t.sumAlongAxisInPlace( a, 1u, true );
	}
	(void)a.max_along_axis( 1u );

	std::mt19937 gen( 42 );
	a.randomize( gen );
	a.randomize( 7u );
	std::mt19937_64 hegen( 99u );
	a.randomizeHe( 64u, hegen() );
}

TEST_CASE( "CudaTensor maxCoeff returns largest absolute value", "[cuda_tensor][maxCoeff]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	CudaTensor<float> t( 2u, 2u, 0.f );
	t.assign( 0, 0, -4.f );
	t.assign( 0, 1, 1.f );
	t.assign( 1, 0, 2.f );
	t.assign( 1, 1, -3.f );
	REQUIRE_THAT( t.maxCoeff(), WithinAbs( 4.f, 1e-5f ) );
}

TEST_CASE( "CudaTensor maxCoeff empty", "[cuda_tensor][maxCoeff]" )
{
	CudaTensor<float> const t;
	REQUIRE( t.maxCoeff() == 0.f );
}

TEST_CASE( "CudaTensor addition is element-wise", "[cuda_tensor][add]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	CudaTensor<float> a( 2u, 2u, 0.f );
	a.assign( 0, 0, 1.f );
	a.assign( 1, 1, 2.f );
	CudaTensor<float> b( 2u, 2u, 0.f );
	b.assign( 0, 1, 3.f );
	b.assign( 1, 0, 4.f );

	CudaTensor<float> c = a + b;
	REQUIRE_THAT( c( 0, 0 ), WithinAbs( 1.f, 1e-5f ) );
	REQUIRE_THAT( c( 0, 1 ), WithinAbs( 3.f, 1e-5f ) );
	REQUIRE_THAT( c( 1, 0 ), WithinAbs( 4.f, 1e-5f ) );
	REQUIRE_THAT( c( 1, 1 ), WithinAbs( 2.f, 1e-5f ) );

	CudaTensor<float> a2( 2u, 2u, 0.f );
	a2.assign( 0, 0, 10.f );
	CudaTensor<float> const b2( 2u, 2u, 1.f );
	a2 += b2;
	REQUIRE_THAT( a2( 0, 0 ), WithinAbs( 11.f, 1e-5f ) );
	REQUIRE_THAT( a2( 1, 1 ), WithinAbs( 1.f, 1e-5f ) );
}

TEST_CASE( "CudaTensor element-wise multiply and scalar scale", "[cuda_tensor][mul]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	CudaTensor<float> a( 2u, 2u, 0.f );
	a.assign( 0, 0, 2.f );
	a.assign( 0, 1, 3.f );
	a.assign( 1, 0, -1.f );
	a.assign( 1, 1, 4.f );
	CudaTensor<float> b( 2u, 2u, 0.f );
	b.assign( 0, 0, 0.5f );
	b.assign( 0, 1, 2.f );
	b.assign( 1, 0, 3.f );
	b.assign( 1, 1, 0.25f );

	CudaTensor<float> c = a * b;
	REQUIRE_THAT( c( 0, 0 ), WithinAbs( 1.f, 1e-5f ) );
	REQUIRE_THAT( c( 0, 1 ), WithinAbs( 6.f, 1e-5f ) );
	REQUIRE_THAT( c( 1, 0 ), WithinAbs( -3.f, 1e-5f ) );
	REQUIRE_THAT( c( 1, 1 ), WithinAbs( 1.f, 1e-5f ) );

	CudaTensor<float> out_wrong( 1u, 1u, 0.f );
	a.elementwiseMultiply( b, out_wrong );
	REQUIRE( out_wrong.rows() == 2u );
	REQUIRE( out_wrong.cols() == 2u );
	REQUIRE_THAT( out_wrong( 0, 0 ), WithinAbs( 1.f, 1e-5f ) );
	REQUIRE_THAT( out_wrong( 1, 1 ), WithinAbs( 1.f, 1e-5f ) );

	CudaTensor<float> a2( 2u, 2u, 2.f );
	CudaTensor<float> const b2( 2u, 2u, 3.f );
	a2 *= b2;
	REQUIRE_THAT( a2( 0, 0 ), WithinAbs( 6.f, 1e-5f ) );

	CudaTensor<float> s = a * 0.5f;
	REQUIRE_THAT( s( 0, 0 ), WithinAbs( 1.f, 1e-5f ) );
	CudaTensor<float> a3( 2u, 2u, 4.f );
	a3 *= 2.f;
	REQUIRE_THAT( a3( 0, 0 ), WithinAbs( 8.f, 1e-5f ) );
}

TEST_CASE( "CudaTensor unary ops and reductions match expectations", "[cuda_tensor][ops]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	CudaTensor<float> a( 2u, 2u, 0.f );
	a.assign( 0, 0, 0.5f );
	a.assign( 0, 1, 2.f );
	a.assign( 1, 0, -1.f );
	a.assign( 1, 1, 4.f );

	CudaTensor<float> g = a.cwiseGreater( 0.f );
	REQUIRE_THAT( g( 0, 0 ), WithinAbs( 1.f, 1e-5f ) );
	REQUIRE_THAT( g( 1, 0 ), WithinAbs( 0.f, 1e-5f ) );

	CudaTensor<float> om = a.cwiseOneMinus();
	REQUIRE_THAT( om( 0, 0 ), WithinAbs( 0.5f, 1e-5f ) );

	CudaTensor<float> e = a.cwiseExp();
	REQUIRE_THAT( e( 0, 1 ), WithinAbs( std::exp( 2.f ), 1e-4f ) );

	CudaTensor<float> lg = a.cwiseLog();
	REQUIRE_THAT( lg( 1, 1 ), WithinAbs( std::log( 4.f ), 1e-4f ) );

	REQUIRE_THAT( a.sum(), WithinAbs( 5.5f, 1e-4f ) );
	REQUIRE_THAT( a.asum(), WithinAbs( 7.5f, 1e-4f ) );

	CudaTensor<float> s0 = a.sumAlongAxis( 0u );
	REQUIRE( s0.rows() == 1u );
	REQUIRE( s0.cols() == 2u );
	REQUIRE_THAT( s0( 0, 0 ), WithinAbs( -0.5f, 1e-4f ) );
	REQUIRE_THAT( s0( 0, 1 ), WithinAbs( 6.f, 1e-4f ) );

	CudaTensor<float> s1 = a.sumAlongAxis( 1u );
	REQUIRE( s1.rows() == 2u );
	REQUIRE( s1.cols() == 1u );
	REQUIRE_THAT( s1( 0, 0 ), WithinAbs( 2.5f, 1e-4f ) );
	REQUIRE_THAT( s1( 1, 0 ), WithinAbs( 3.f, 1e-4f ) );

	CudaTensor<float> s0t = a.sumAlongAxis( 0u, true );
	REQUIRE( s0t.rows() == 2u );
	REQUIRE( s0t.cols() == 1u );
	REQUIRE_THAT( s0t( 0, 0 ), WithinAbs( -0.5f, 1e-4f ) );
	REQUIRE_THAT( s0t( 1, 0 ), WithinAbs( 6.f, 1e-4f ) );

	CudaTensor<float> s1t = a.sumAlongAxis( 1u, true );
	REQUIRE( s1t.rows() == 1u );
	REQUIRE( s1t.cols() == 2u );
	REQUIRE_THAT( s1t( 0, 0 ), WithinAbs( 2.5f, 1e-4f ) );
	REQUIRE_THAT( s1t( 0, 1 ), WithinAbs( 3.f, 1e-4f ) );

	CudaTensor<float> m0 = a.max_along_axis( 0u );
	REQUIRE_THAT( m0( 0, 0 ), WithinAbs( 0.5f, 1e-4f ) );
	REQUIRE_THAT( m0( 0, 1 ), WithinAbs( 4.f, 1e-4f ) );

	CudaTensor<float> m1 = a.max_along_axis( 1u );
	REQUIRE_THAT( m1( 0, 0 ), WithinAbs( 2.f, 1e-4f ) );
	REQUIRE_THAT( m1( 1, 0 ), WithinAbs( 4.f, 1e-4f ) );

	std::vector<float> buf( 4u );
#if NEURAL_CUDA_ENABLED
	a.copyToHost( buf.data() );
#else
	std::copy_n( a.data(), a.size(), buf.begin() );
#endif
	REQUIRE_THAT( buf[0], WithinAbs( 0.5f, 1e-5f ) );
	REQUIRE_THAT( buf[3], WithinAbs( 4.f, 1e-5f ) );
}

TEST_CASE( "CudaTensor transpose reorders elements (matches Eigen)", "[cuda_tensor][transpose]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	// 2×2 row-major: [[1,2],[3,4]] → transpose [[1,3],[2,4]]
	CudaTensor<float> a( 2u, 2u, 0.f );
	a.assign( 0, 0, 1.f );
	a.assign( 0, 1, 2.f );
	a.assign( 1, 0, 3.f );
	a.assign( 1, 1, 4.f );
	CudaTensor<float> t = a.transpose();
	REQUIRE( t.rows() == 2u );
	REQUIRE( t.cols() == 2u );
	REQUIRE_THAT( t( 0, 0 ), WithinAbs( 1.f, 1e-5f ) );
	REQUIRE_THAT( t( 0, 1 ), WithinAbs( 3.f, 1e-5f ) );
	REQUIRE_THAT( t( 1, 0 ), WithinAbs( 2.f, 1e-5f ) );
	REQUIRE_THAT( t( 1, 1 ), WithinAbs( 4.f, 1e-5f ) );
}

TEST_CASE( "CudaTensor assignTensorAsRow copies device row with cudaMemcpy", "[cuda_tensor][assignTensorAsRow]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::vector<float> const r0 = { 1.f, 2.f, 3.f };
	std::vector<float> const r1 = { 4.f, 5.f, 6.f };
	CudaTensor<float> row0( 1u, 3u, r0.begin(), r0.end() );
	CudaTensor<float> row1( 1u, 3u, r1.begin(), r1.end() );
	CudaTensor<float> batch( 2u, 3u, 0.f );
	batch.assignTensorAsRow( 0u, row0 );
	batch.assignTensorAsRow( 1u, row1 );
	REQUIRE_THAT( batch( 0, 0 ), WithinAbs( 1.f, 1e-5f ) );
	REQUIRE_THAT( batch( 0, 2 ), WithinAbs( 3.f, 1e-5f ) );
	REQUIRE_THAT( batch( 1, 0 ), WithinAbs( 4.f, 1e-5f ) );
	REQUIRE_THAT( batch( 1, 2 ), WithinAbs( 6.f, 1e-5f ) );
}

TEST_CASE( "CudaTensor assignTensorBlock gathers full rows like Tensor", "[cuda_tensor][assignTensorBlock]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	CudaTensor<float> src( 3u, 5u, 0.f );
	for ( std::size_t row = 0; row < 3u; ++row ) {
		for ( std::size_t c = 0; c < 5u; ++c ) {
			src.assign( row, c, static_cast<float>( row + 1 ) );
		}
	}
	CudaTensor<float> dst( 2u, 5u, 0.f );
	std::vector<int> const indices{ 2, 0 };
	dst.assignTensorBlock( src, indices, 0u, 2u );
	REQUIRE_THAT( dst( 0, 0 ), WithinAbs( 3.f, 1e-5f ) );
	REQUIRE_THAT( dst( 1, 4 ), WithinAbs( 1.f, 1e-5f ) );
}

TEST_CASE( "CudaTensor assignTensorBlock row_indices_src window", "[cuda_tensor][assignTensorBlock]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	CudaTensor<float> src( 3u, 4u, 0.f );
	for ( std::size_t row = 0; row < 3u; ++row ) {
		for ( std::size_t c = 0; c < 4u; ++c ) {
			src.assign( row, c, static_cast<float>( 10 * row + static_cast<int>( c ) ) );
		}
	}
	CudaTensor<float> dst( 1u, 4u, 0.f );
	std::vector<int> const indices{ 2, 0, 1 };
	dst.assignTensorBlock( src, indices, 1u, 1u );
	REQUIRE_THAT( dst( 0, 0 ), WithinAbs( 0.f, 1e-5f ) );
	REQUIRE_THAT( dst( 0, 3 ), WithinAbs( 3.f, 1e-5f ) );
}

TEST_CASE( "CudaTensor copy and move preserve shape", "[cuda_tensor][rule_of_five]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	CudaTensor<float> a( 4u, 5u );
	CudaTensor<float> b( a );
	REQUIRE( b.rows() == 4u );
	REQUIRE( b.cols() == 5u );

	CudaTensor<float> c( std::move( b ) );
	REQUIRE( c.rows() == 4u );
	REQUIRE( c.cols() == 5u );

	CudaTensor<float> d;
	d = c;
	REQUIRE( d.rows() == 4u );
	REQUIRE( d.cols() == 5u );
}
