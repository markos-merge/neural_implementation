#include "tensor.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

using neural::Tensor;
using Catch::Matchers::WithinAbs;

TEST_CASE( "Tensor assignTensorBlock gathers full rows by source row index (std::vector<int>)", "[tensor][assignTensorBlock]" )
{
	// Three source rows: row i filled with float(i+1).
	Tensor<float> src( 3, 5, 0.f );
	for ( std::size_t row = 0; row < 3u; ++row ) {
		for ( std::size_t c = 0; c < 5u; ++c ) {
			src.assign( row, c, static_cast<float>( row + 1 ) );
		}
	}
	Tensor<float> dst( 2, 5, 0.f );
	std::vector<int> const indices{ 2, 0 };
	dst.assignTensorBlock( src, indices, 0u, 2u );
	REQUIRE_THAT( dst( 0, 0 ), WithinAbs( 3.f, 1e-6f ) );
	REQUIRE_THAT( dst( 1, 4 ), WithinAbs( 1.f, 1e-6f ) );
}

TEST_CASE( "Tensor assignTensorBlock respects row_indices_src window into indices", "[tensor][assignTensorBlock]" )
{
	Tensor<float> src( 3, 4, 0.f );
	for ( std::size_t row = 0; row < 3u; ++row ) {
		for ( std::size_t c = 0; c < 4u; ++c ) {
			src.assign( row, c, static_cast<float>( 10 * row + static_cast<int>( c ) ) );
		}
	}
	Tensor<float> dst( 1, 4, 0.f );
	std::vector<int> const indices{ 2, 0, 1 };
	dst.assignTensorBlock( src, indices, 1u, 1u );
	REQUIRE_THAT( dst( 0, 0 ), WithinAbs( 0.f, 1e-6f ) );
	REQUIRE_THAT( dst( 0, 3 ), WithinAbs( 3.f, 1e-6f ) );
}

TEST_CASE( "Tensor assignTensorBlock row_indices_size zero is no-op", "[tensor][assignTensorBlock]" )
{
	Tensor<float> src( 1, 3, 1.f );
	Tensor<float> dst( 1, 3, 0.f );
	std::vector<int> const indices{ 0 };
	dst.assignTensorBlock( src, indices, 0u, 0u );
	REQUIRE_THAT( dst( 0, 0 ), WithinAbs( 0.f, 1e-6f ) );
}

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
