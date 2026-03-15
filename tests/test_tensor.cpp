#include "tensor.hpp"
#include "tensor_like.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <random>
#include <vector>

using neural::Tensor;
static_assert( TensorLike<Tensor<float>> );

using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps = 1e-5f;

Tensor<float> make_from_vec( std::size_t rows, std::size_t cols,
                             std::vector<float> const &data )
{
	return Tensor<float>( rows, cols, data.begin(), data.end() );
}
} // namespace

TEST_CASE( "Tensor shape", "[tensor][shape]" )
{
	SECTION( "default constructor gives 0x0" )
	{
		Tensor<float> t;
		REQUIRE( t.rows() == 0u );
		REQUIRE( t.cols() == 0u );
		REQUIRE( t.size() == 0u );
	}

	SECTION( "(rows, cols) constructor" )
	{
		Tensor<float> t( 2, 3 );
		REQUIRE( t.rows() == 2u );
		REQUIRE( t.cols() == 3u );
		REQUIRE( t.size() == 6u );
	}

	SECTION( "(rows, cols, value) initializes and shape is correct" )
	{
		Tensor<float> t( 3, 4, 1.0f );
		REQUIRE( t.rows() == 3u );
		REQUIRE( t.cols() == 4u );
		REQUIRE( t.size() == 12u );
	}

	SECTION( "construction from range gives correct shape" )
	{
		std::vector<float> data( 6, 1.0f );
		Tensor<float> t( 2, 3, data.begin(), data.end() );
		REQUIRE( t.rows() == 2u );
		REQUIRE( t.cols() == 3u );
		REQUIRE( t.size() == 6u );
	}
}

TEST_CASE( "Tensor element access and construction from data",
           "[tensor][access]" )
{
	std::vector<float> data = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
	Tensor<float> t( 2, 3, data.begin(), data.end() );

	REQUIRE_THAT( t( 0, 0 ), WithinAbs( 1.0f, eps ) );
	REQUIRE_THAT( t( 0, 1 ), WithinAbs( 2.0f, eps ) );
	REQUIRE_THAT( t( 0, 2 ), WithinAbs( 3.0f, eps ) );
	REQUIRE_THAT( t( 1, 0 ), WithinAbs( 4.0f, eps ) );
	REQUIRE_THAT( t( 1, 1 ), WithinAbs( 5.0f, eps ) );
	REQUIRE_THAT( t( 1, 2 ), WithinAbs( 6.0f, eps ) );
}

TEST_CASE( "Tensor matmul", "[tensor][matmul]" )
{
	// A 2x3, B 3x2 -> C 2x2
	// A = [1 2 3; 4 5 6], B = [1 2; 3 4; 5 6]
	// C(0,0)=1+6+15=22, C(0,1)=2+8+18=28, C(1,0)=4+15+30=49, C(1,1)=8+20+36=64
	std::vector<float> a_data = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
	std::vector<float> b_data = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
	Tensor<float> A( 2, 3, a_data.begin(), a_data.end() );
	Tensor<float> B( 3, 2, b_data.begin(), b_data.end() );

	Tensor<float> C = A.matmul( B );

	REQUIRE( C.rows() == 2u );
	REQUIRE( C.cols() == 2u );
	REQUIRE_THAT( C( 0, 0 ), WithinAbs( 22.0f, eps ) );
	REQUIRE_THAT( C( 0, 1 ), WithinAbs( 28.0f, eps ) );
	REQUIRE_THAT( C( 1, 0 ), WithinAbs( 49.0f, eps ) );
	REQUIRE_THAT( C( 1, 1 ), WithinAbs( 64.0f, eps ) );

	SECTION( "matmulInPlace() modifies tensor" )
	{
		Tensor<float> D = A;
		D.matmulInPlace( B );
		REQUIRE( D.rows() == 2u );
		REQUIRE( D.cols() == 2u );
		REQUIRE_THAT( D( 0, 0 ), WithinAbs( 22.0f, eps ) );
		REQUIRE_THAT( D( 0, 1 ), WithinAbs( 28.0f, eps ) );
		REQUIRE_THAT( D( 1, 0 ), WithinAbs( 49.0f, eps ) );
		REQUIRE_THAT( D( 1, 1 ), WithinAbs( 64.0f, eps ) );
	}
}

TEST_CASE( "Tensor transpose", "[tensor][transpose]" )
{
	std::vector<float> data = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
	Tensor<float> A( 2, 3, data.begin(), data.end() );
	Tensor<float> B = A;
	B.transposeInPlace();

	REQUIRE( B.rows() == 3u );
	REQUIRE( B.cols() == 2u );
	REQUIRE_THAT( B( 0, 0 ), WithinAbs( 1.0f, eps ) );
	REQUIRE_THAT( B( 0, 1 ), WithinAbs( 4.0f, eps ) );
	REQUIRE_THAT( B( 1, 0 ), WithinAbs( 2.0f, eps ) );
	REQUIRE_THAT( B( 1, 1 ), WithinAbs( 5.0f, eps ) );
	REQUIRE_THAT( B( 2, 0 ), WithinAbs( 3.0f, eps ) );
	REQUIRE_THAT( B( 2, 1 ), WithinAbs( 6.0f, eps ) );

	SECTION( "transpose() returns new tensor, does not modify original" )
	{
		Tensor<float> C = A.transpose();
		REQUIRE( C.rows() == 3u );
		REQUIRE( C.cols() == 2u );
		REQUIRE( A.rows() == 2u );
		REQUIRE( A.cols() == 3u );
		REQUIRE_THAT( C( 0, 0 ), WithinAbs( 1.0f, eps ) );
		REQUIRE_THAT( C( 0, 1 ), WithinAbs( 4.0f, eps ) );
	}

	SECTION( "(A^T)^T equals A" )
	{
		Tensor<float> At = A;
		At.transposeInPlace();
		At.transposeInPlace();
		REQUIRE( At.rows() == A.rows() );
		REQUIRE( At.cols() == A.cols() );
		for ( std::size_t i = 0; i < A.rows(); ++i )
			for ( std::size_t j = 0; j < A.cols(); ++j )
				REQUIRE_THAT( At( i, j ), WithinAbs( A( i, j ), eps ) );
	}
}

TEST_CASE( "Tensor reshape", "[tensor][reshape]" )
{
	// A = [1 2 3; 4 5 6], Tensor uses RowMajor: [1 2; 3 4; 5 6] when reshaped to
	// 3x2
	std::vector<float> data = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
	Tensor<float> A( 2, 3, data.begin(), data.end() );

	SECTION( "reshape() returns new tensor, does not modify original" )
	{
		Tensor<float> B = A.reshape( 3, 2 );
		REQUIRE( B.rows() == 3u );
		REQUIRE( B.cols() == 2u );
		REQUIRE( A.rows() == 2u );
		REQUIRE( A.cols() == 3u );
		// RowMajor order: [1 2; 3 4; 5 6]
		REQUIRE_THAT( B( 0, 0 ), WithinAbs( 1.0f, eps ) );
		REQUIRE_THAT( B( 0, 1 ), WithinAbs( 2.0f, eps ) );
		REQUIRE_THAT( B( 1, 0 ), WithinAbs( 3.0f, eps ) );
		REQUIRE_THAT( B( 1, 1 ), WithinAbs( 4.0f, eps ) );
		REQUIRE_THAT( B( 2, 0 ), WithinAbs( 5.0f, eps ) );
		REQUIRE_THAT( B( 2, 1 ), WithinAbs( 6.0f, eps ) );
	}

	SECTION( "reshapeInPlace() modifies tensor, same layout as reshape()" )
	{
		Tensor<float> C = A;
		C.reshapeInPlace( 3, 2 );
		REQUIRE( C.rows() == 3u );
		REQUIRE( C.cols() == 2u );
		// RowMajor order: [1 2; 3 4; 5 6]
		REQUIRE_THAT( C( 0, 0 ), WithinAbs( 1.0f, eps ) );
		REQUIRE_THAT( C( 0, 1 ), WithinAbs( 2.0f, eps ) );
		REQUIRE_THAT( C( 1, 0 ), WithinAbs( 3.0f, eps ) );
		REQUIRE_THAT( C( 1, 1 ), WithinAbs( 4.0f, eps ) );
		REQUIRE_THAT( C( 2, 0 ), WithinAbs( 5.0f, eps ) );
		REQUIRE_THAT( C( 2, 1 ), WithinAbs( 6.0f, eps ) );
	}

	SECTION( "reshape to 1x6" )
	{
		Tensor<float> D = A.reshape( 1, 6 );
		REQUIRE( D.rows() == 1u );
		REQUIRE( D.cols() == 6u );
		// RowMajor: 1,2,3,4,5,6
		for ( std::size_t j = 0; j < 6u; ++j )
			REQUIRE_THAT( D( 0, j ), WithinAbs( static_cast<float>( j + 1 ), eps ) );
	}
}

TEST_CASE( "Tensor element-wise add", "[tensor][elementwise]" )
{
	std::vector<float> a_data = { 1.f, 2.f, 3.f, 4.f };
	std::vector<float> b_data = { 0.f, 1.f, 2.f, 3.f };
	Tensor<float> A( 2, 2, a_data.begin(), a_data.end() );
	Tensor<float> B( 2, 2, b_data.begin(), b_data.end() );

	Tensor<float> C = A + B;

	REQUIRE( C.rows() == 2u );
	REQUIRE( C.cols() == 2u );
	REQUIRE_THAT( C( 0, 0 ), WithinAbs( 1.0f, eps ) );
	REQUIRE_THAT( C( 0, 1 ), WithinAbs( 3.0f, eps ) );
	REQUIRE_THAT( C( 1, 0 ), WithinAbs( 5.0f, eps ) ); // 3 + 2
	REQUIRE_THAT( C( 1, 1 ), WithinAbs( 7.0f, eps ) );
}

TEST_CASE( "Tensor operator+=", "[tensor][elementwise]" )
{
	std::vector<float> a_data = { 1.f, 2.f, 3.f, 4.f };
	std::vector<float> b_data = { 0.f, 1.f, 2.f, 3.f };
	Tensor<float> A( 2, 2, a_data.begin(), a_data.end() );
	Tensor<float> B( 2, 2, b_data.begin(), b_data.end() );

	A += B;

	REQUIRE_THAT( A( 0, 0 ), WithinAbs( 1.0f, eps ) );
	REQUIRE_THAT( A( 0, 1 ), WithinAbs( 3.0f, eps ) );
	REQUIRE_THAT( A( 1, 0 ), WithinAbs( 5.0f, eps ) );
	REQUIRE_THAT( A( 1, 1 ), WithinAbs( 7.0f, eps ) );
}

TEST_CASE( "Tensor element-wise multiply", "[tensor][elementwise]" )
{
	std::vector<float> a_data = { 1.f, 2.f, 3.f, 4.f };
	std::vector<float> b_data = { 2.f, 3.f, 4.f, 5.f };
	Tensor<float> A( 2, 2, a_data.begin(), a_data.end() );
	Tensor<float> B( 2, 2, b_data.begin(), b_data.end() );

	Tensor<float> C = A * B;

	REQUIRE( C.rows() == 2u );
	REQUIRE( C.cols() == 2u );
	REQUIRE_THAT( C( 0, 0 ), WithinAbs( 2.0f, eps ) );
	REQUIRE_THAT( C( 0, 1 ), WithinAbs( 6.0f, eps ) );
	REQUIRE_THAT( C( 1, 0 ), WithinAbs( 12.0f, eps ) );
	REQUIRE_THAT( C( 1, 1 ), WithinAbs( 20.0f, eps ) );
}

TEST_CASE( "Tensor operator*=", "[tensor][elementwise]" )
{
	std::vector<float> a_data = { 1.f, 2.f, 3.f, 4.f };
	std::vector<float> b_data = { 2.f, 2.f, 2.f, 2.f };
	Tensor<float> A( 2, 2, a_data.begin(), a_data.end() );
	Tensor<float> B( 2, 2, b_data.begin(), b_data.end() );

	A *= B;

	REQUIRE_THAT( A( 0, 0 ), WithinAbs( 2.0f, eps ) );
	REQUIRE_THAT( A( 0, 1 ), WithinAbs( 4.0f, eps ) );
	REQUIRE_THAT( A( 1, 0 ), WithinAbs( 6.0f, eps ) );
	REQUIRE_THAT( A( 1, 1 ), WithinAbs( 8.0f, eps ) );
}

TEST_CASE( "Tensor scalar multiply", "[tensor][scalar]" )
{
	std::vector<float> data = { 1.f, 2.f, 3.f, 4.f };
	Tensor<float> A( 2, 2, data.begin(), data.end() );

	Tensor<float> B = A * 2.0f;

	REQUIRE_THAT( B( 0, 0 ), WithinAbs( 2.0f, eps ) );
	REQUIRE_THAT( B( 0, 1 ), WithinAbs( 4.0f, eps ) );
	REQUIRE_THAT( B( 1, 0 ), WithinAbs( 6.0f, eps ) );
	REQUIRE_THAT( B( 1, 1 ), WithinAbs( 8.0f, eps ) );
}

TEST_CASE( "Tensor operator*=(scalar)", "[tensor][scalar]" )
{
	std::vector<float> data = { 1.f, 2.f, 3.f, 4.f };
	Tensor<float> A( 2, 2, data.begin(), data.end() );

	A *= 2.0f;

	REQUIRE_THAT( A( 0, 0 ), WithinAbs( 2.0f, eps ) );
	REQUIRE_THAT( A( 0, 1 ), WithinAbs( 4.0f, eps ) );
	REQUIRE_THAT( A( 1, 0 ), WithinAbs( 6.0f, eps ) );
	REQUIRE_THAT( A( 1, 1 ), WithinAbs( 8.0f, eps ) );
}

TEST_CASE( "Tensor cwiseGreater", "[tensor][elementwise]" )
{
	// mask = 1 where x > threshold, 0 elsewhere
	std::vector<float> data = { -1.f, 0.f, 1.f, 2.f };
	Tensor<float> A( 2, 2, data.begin(), data.end() );

	Tensor<float> mask = A.cwiseGreater( 0.f );

	REQUIRE( mask.rows() == 2u );
	REQUIRE( mask.cols() == 2u );
	REQUIRE_THAT( mask( 0, 0 ), WithinAbs( 0.0f, eps ) ); // -1 <= 0
	REQUIRE_THAT( mask( 0, 1 ), WithinAbs( 0.0f, eps ) ); // 0 not > 0
	REQUIRE_THAT( mask( 1, 0 ), WithinAbs( 1.0f, eps ) ); // 1 > 0
	REQUIRE_THAT( mask( 1, 1 ), WithinAbs( 1.0f, eps ) ); // 2 > 0

	SECTION( "cwiseGreater(1) gives 1 only for elements > 1" )
	{
		Tensor<float> m = A.cwiseGreater( 1.f );
		REQUIRE_THAT( m( 0, 0 ), WithinAbs( 0.0f, eps ) );
		REQUIRE_THAT( m( 0, 1 ), WithinAbs( 0.0f, eps ) );
		REQUIRE_THAT( m( 1, 0 ), WithinAbs( 0.0f, eps ) );
		REQUIRE_THAT( m( 1, 1 ), WithinAbs( 1.0f, eps ) );
	}
}

TEST_CASE( "Tensor sum", "[tensor][sum]" )
{
	std::vector<float> data = { 1.f, 2.f, 3.f, 4.f };
	Tensor<float> A( 2, 2, data.begin(), data.end() );

	REQUIRE_THAT( A.sum(), WithinAbs( 10.0f, eps ) );
}

TEST_CASE( "Tensor sum_along_axis", "[tensor][sum_along_axis]" )
{
	// [1 2 3; 4 5 6]
	// axis 0: sum cols -> [5 7 9]   shape (1, 3)
	// axis 1: sum rows -> [6; 15]   shape (2, 1)
	std::vector<float> data = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
	Tensor<float> A( 2, 3, data.begin(), data.end() );

	Tensor<float> col_sums = A.sum_along_axis( 0 );
	REQUIRE( col_sums.rows() == 1u );
	REQUIRE( col_sums.cols() == 3u );
	REQUIRE_THAT( col_sums( 0, 0 ), WithinAbs( 5.0f, eps ) );
	REQUIRE_THAT( col_sums( 0, 1 ), WithinAbs( 7.0f, eps ) );
	REQUIRE_THAT( col_sums( 0, 2 ), WithinAbs( 9.0f, eps ) );

	Tensor<float> row_sums = A.sum_along_axis( 1 );
	REQUIRE( row_sums.rows() == 2u );
	REQUIRE( row_sums.cols() == 1u );
	REQUIRE_THAT( row_sums( 0, 0 ), WithinAbs( 6.0f, eps ) );
	REQUIRE_THAT( row_sums( 1, 0 ), WithinAbs( 15.0f, eps ) );
}

TEST_CASE( "Tensor (rows, cols) zeros", "[tensor][construction]" )
{
	Tensor<float> t( 2, 3 );
	for ( std::size_t i = 0; i < t.rows(); ++i )
		for ( std::size_t j = 0; j < t.cols(); ++j )
			REQUIRE_THAT( t( i, j ), WithinAbs( 0.0f, eps ) );
}

TEST_CASE( "Tensor (rows, cols, value) constant", "[tensor][construction]" )
{
	Tensor<float> t( 2, 3, 3.14f );
	for ( std::size_t i = 0; i < t.rows(); ++i )
		for ( std::size_t j = 0; j < t.cols(); ++j )
			REQUIRE_THAT( t( i, j ), WithinAbs( 3.14f, eps ) );
}

TEST_CASE( "Tensor invalid range size throws", "[tensor][construction]" )
{
	std::vector<float> data = { 1.f, 2.f, 3.f }; // size 3
	REQUIRE_THROWS_AS( Tensor<float>( 2, 2, data.begin(), data.end() ),
	                   std::invalid_argument );
}

TEST_CASE( "Tensor randomize with generator", "[tensor][randomize]" )
{
	Tensor<float> t( 2, 3 );
	std::mt19937 gen( 42 );
	std::uniform_real_distribution<float> dis( 0.0f, 1.0f );
	auto generator = [&]() { return dis( gen ); };
	t.randomize( generator );

	SECTION( "all elements are within [0, 1]" )
	{
		for ( std::size_t i = 0; i < t.rows(); ++i )
			for ( std::size_t j = 0; j < t.cols(); ++j ) {
				REQUIRE( t( i, j ) >= 0.0f );
				REQUIRE( t( i, j ) <= 1.0f );
			}
	}

	SECTION( "deterministic with fixed seed produces same values" )
	{
		Tensor<float> a( 2, 3 );
		Tensor<float> b( 2, 3 );
		std::mt19937 gen_a( 123 );
		std::mt19937 gen_b( 123 );
		std::uniform_real_distribution<float> dis_a( 0.0f, 1.0f );
		std::uniform_real_distribution<float> dis_b( 0.0f, 1.0f );
		auto gen_fn_a = [&]() { return dis_a( gen_a ); };
		auto gen_fn_b = [&]() { return dis_b( gen_b ); };
		a.randomize( gen_fn_a );
		b.randomize( gen_fn_b );
		for ( std::size_t i = 0; i < a.rows(); ++i )
			for ( std::size_t j = 0; j < a.cols(); ++j )
				REQUIRE_THAT( a( i, j ), WithinAbs( b( i, j ), eps ) );
	}

	SECTION( "custom range is respected" )
	{
		Tensor<float> t2( 3, 2 );
		std::mt19937 gen2( 0 );
		std::uniform_real_distribution<float> dis2( -1.0f, 1.0f );
		auto gen_fn = [&]() { return dis2( gen2 ); };
		t2.randomize( gen_fn );
		for ( std::size_t i = 0; i < t2.rows(); ++i )
			for ( std::size_t j = 0; j < t2.cols(); ++j ) {
				REQUIRE( t2( i, j ) >= -1.0f );
				REQUIRE( t2( i, j ) <= 1.0f );
			}
	}
}

TEST_CASE( "Tensor randomize default", "[tensor][randomize]" )
{
	Tensor<float> t( 4, 5 );
	t.randomize();

	SECTION( "all elements are within [0, 1]" )
	{
		for ( std::size_t i = 0; i < t.rows(); ++i )
			for ( std::size_t j = 0; j < t.cols(); ++j ) {
				REQUIRE( t( i, j ) >= 0.0f );
				REQUIRE( t( i, j ) <= 1.0f );
			}
	}

	SECTION( "elements are not all identical" )
	{
		float first = t( 0, 0 );
		bool found_different = false;
		for ( std::size_t i = 0; i < t.rows() && !found_different; ++i )
			for ( std::size_t j = 0; j < t.cols(); ++j )
				if ( std::abs( t( i, j ) - first ) > eps ) {
					found_different = true;
					break;
				}
		REQUIRE( found_different );
	}
}
