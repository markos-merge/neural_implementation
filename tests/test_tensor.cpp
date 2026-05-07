#include "tensor.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

using neural::Tensor;
using Catch::Matchers::WithinAbs;

TEST_CASE( "Tensor maxCoeff is largest absolute value", "[tensor][maxCoeff]" )
{
	std::vector<float> const data = { 1.f, -0.5f, -2.f };
	Tensor<float> const t( 1, 3, data.begin(), data.end() );
	REQUIRE_THAT( t.maxCoeff(), WithinAbs( 2.f, 1e-6f ) );
}

TEST_CASE( "Tensor maxCoeff prefers magnitude over algebraic max", "[tensor][maxCoeff]" )
{
	std::vector<float> const data = { -3.f, 1.f };
	Tensor<float> const t( 1, 2, data.begin(), data.end() );
	REQUIRE_THAT( t.maxCoeff(), WithinAbs( 3.f, 1e-6f ) );
}

TEST_CASE( "Tensor maxCoeff empty tensor", "[tensor][maxCoeff]" )
{
	Tensor<float> const t;
	REQUIRE( t.maxCoeff() == 0.f );
}

TEST_CASE( "Tensor asum matches sum of absolute values", "[tensor][asum]" )
{
	std::vector<float> const data = { 1.f, -2.f, 3.f };
	Tensor<float> const t( 1, 3, data.begin(), data.end() );
	REQUIRE_THAT( t.asum(), WithinAbs( 6.f, 1e-6f ) );
}

TEST_CASE( "Tensor assign scalar fills all elements", "[tensor][assign]" )
{
	Tensor<float> t( 2, 2, 1.f );
	t.assign( 0.f );
	REQUIRE_THAT( t( 0, 0 ), WithinAbs( 0.f, 1e-6f ) );
	REQUIRE_THAT( t( 1, 1 ), WithinAbs( 0.f, 1e-6f ) );
}

TEST_CASE( "Tensor elementwiseMultiply into output resizes and matches operator*", "[tensor][mul]" )
{
	Tensor<float> a( 2, 2, 2.f );
	Tensor<float> b( 2, 2, 3.f );
	Tensor<float> out( 1, 1, 0.f );
	a.elementwiseMultiply( b, out );
	Tensor<float> const expected = a * b;
	REQUIRE( out.rows() == 2u );
	REQUIRE( out.cols() == 2u );
	REQUIRE_THAT( out( 0, 0 ), WithinAbs( expected( 0, 0 ), 1e-6f ) );
	REQUIRE_THAT( out( 1, 1 ), WithinAbs( expected( 1, 1 ), 1e-6f ) );
}

TEST_CASE( "Tensor non-owning elementwiseMultiply throws if out would resize", "[tensor][view]" )
{
	std::vector<float> buf_a( 4, 2.f );
	std::vector<float> buf_b( 4, 3.f );
	std::vector<float> buf_out( 1, 0.f );
	Tensor<float> a( buf_a.data(), 2, 2, false );
	Tensor<float> b( buf_b.data(), 2, 2, false );
	Tensor<float> out( buf_out.data(), 1, 1, false );
	REQUIRE_THROWS_AS( a.elementwiseMultiply( b, out ), std::invalid_argument );
}

TEST_CASE( "Tensor non-owning move-assign throws", "[tensor][view]" )
{
	std::vector<float> buf( 4 );
	Tensor<float> v( buf.data(), 2, 2, false );
	Tensor<float> rhs( 2, 2, 1.f );
	REQUIRE_THROWS_AS( v = std::move( rhs ), std::invalid_argument );
}

TEST_CASE( "Tensor non-owning copy-assign requires matching shape", "[tensor][view]" )
{
	std::vector<float> buf( 4 );
	Tensor<float> v( buf.data(), 2, 2, false );
	Tensor<float> rhs( 1, 4, 1.f );
	REQUIRE_THROWS_AS( v = rhs, std::invalid_argument );
}

TEST_CASE( "Tensor non-owning copy-assign same shape writes into buffer", "[tensor][view]" )
{
	std::vector<float> buf( 4, 0.f );
	Tensor<float> v( buf.data(), 2, 2, false );
	Tensor<float> rhs( 2, 2, 5.f );
	v = rhs;
	REQUIRE_THAT( buf[0], WithinAbs( 5.f, 1e-6f ) );
}

TEST_CASE( "Tensor pointer ctor with own=false is a non-owning view", "[tensor][view]" )
{
	std::vector<float> buf = { 1.f, 2.f, 3.f, 4.f };
	Tensor<float> t( buf.data(), 2, 2, false );
	REQUIRE_THAT( t( 0, 1 ), WithinAbs( 2.f, 1e-6f ) );
	buf[1] = 9.f;
	REQUIRE_THAT( t( 0, 1 ), WithinAbs( 9.f, 1e-6f ) );
}

TEST_CASE( "Tensor pointer ctor with own=true takes ownership (destructor frees)", "[tensor][own]" )
{
	float *const heap = new float[4]{ 1.f, 2.f, 3.f, 4.f };
	{
		Tensor<float> t( heap, 2, 2, true );
		REQUIRE_THAT( t( 1, 1 ), WithinAbs( 4.f, 1e-6f ) );
	}
	// heap freed by ~Tensor; no double-delete check beyond surviving scope
	Tensor<float> fresh( 1, 1, 0.f );
	REQUIRE_THAT( fresh( 0, 0 ), WithinAbs( 0.f, 1e-6f ) );
}
