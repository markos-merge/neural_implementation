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
