#include "layer.hpp"
#include "layer_wiring_helpers.hpp"
#include "softmax_layer.hpp"
#include "tensor.hpp"
#include "tensor_like.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

using neural::SoftmaxLayer;
using neural::Tensor;
using neural::test::wire_layer;

static_assert( TensorLike<Tensor<float>> );
static_assert( LayerLike<SoftmaxLayer<Tensor<float>>, Tensor<float>> );

using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps = 1e-4f;

float softmax_ref_row( std::vector<float> const &logits, std::size_t j )
{
	double s = 0.0;
	for ( float x : logits )
		s += std::exp( static_cast<double>( x ) );
	return static_cast<float>( std::exp( static_cast<double>( logits.at( j ) ) ) / s );
}

} // namespace

TEST_CASE( "SoftmaxLayer forward output shape", "[softmax_layer][forward][shape]" )
{
	SoftmaxLayer<Tensor<float>> layer;
	Tensor<float> in_buf( 3, 7, 1.0f );
	Tensor<float> out_buf( 3, 7 );
	Tensor<float> g_out( 3, 7 );
	Tensor<float> g_in( 3, 7 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	REQUIRE( out_buf.rows() == 3u );
	REQUIRE( out_buf.cols() == 7u );
}

TEST_CASE( "SoftmaxLayer forward rows sum to one", "[softmax_layer][forward][numerical]" )
{
	SoftmaxLayer<Tensor<float>> layer;

	std::vector<float> data = { 0.f, 1.f, -0.5f, 2.f, 0.f, 0.25f, -1.f, 3.f };
	Tensor<float> in_buf( 2, 4, data.begin(), data.end() );
	Tensor<float> out_buf( 2, 4 );
	Tensor<float> g_out( 2, 4 );
	Tensor<float> g_in( 2, 4 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	for ( std::size_t i = 0; i < out_buf.rows(); ++i ) {
		float row_sum = 0.f;
		for ( std::size_t j = 0; j < out_buf.cols(); ++j )
			row_sum += out_buf( i, j );
		REQUIRE_THAT( row_sum, WithinAbs( 1.0f, eps ) );
	}
}

TEST_CASE( "SoftmaxLayer forward values match reference (single row)", "[softmax_layer][forward][numerical]" )
{
	SoftmaxLayer<Tensor<float>> layer;

	std::vector<float> logits = { 1.f, 2.f, 3.f };
	Tensor<float> in_buf( 1, 3, logits.begin(), logits.end() );
	Tensor<float> out_buf( 1, 3 );
	Tensor<float> g_out( 1, 3 );
	Tensor<float> g_in( 1, 3 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	for ( std::size_t j = 0; j < 3u; ++j )
		REQUIRE_THAT( out_buf( 0, j ), WithinAbs( softmax_ref_row( logits, j ), eps ) );
}

TEST_CASE( "SoftmaxLayer forward equal logits are uniform", "[softmax_layer][forward][numerical]" )
{
	SoftmaxLayer<Tensor<float>> layer;

	Tensor<float> in_buf( 2, 4, 0.0f );
	Tensor<float> out_buf( 2, 4 );
	Tensor<float> g_out( 2, 4 );
	Tensor<float> g_in( 2, 4 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	float const u = 1.0f / 4.0f;
	for ( std::size_t i = 0; i < out_buf.rows(); ++i )
		for ( std::size_t j = 0; j < out_buf.cols(); ++j )
			REQUIRE_THAT( out_buf( i, j ), WithinAbs( u, eps ) );
}

TEST_CASE( "SoftmaxLayer forward one dominant logit", "[softmax_layer][forward][numerical]" )
{
	SoftmaxLayer<Tensor<float>> layer;

	std::vector<float> data = { -5.f, 5.f, -5.f };
	Tensor<float> in_buf( 1, 3, data.begin(), data.end() );
	Tensor<float> out_buf( 1, 3 );
	Tensor<float> g_out( 1, 3 );
	Tensor<float> g_in( 1, 3 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	REQUIRE( out_buf( 0, 1 ) > 0.99f );
	REQUIRE( out_buf( 0, 0 ) < 0.01f );
	REQUIRE( out_buf( 0, 2 ) < 0.01f );
}

TEST_CASE( "SoftmaxLayer backward gradient shape", "[softmax_layer][backward][shape]" )
{
	SoftmaxLayer<Tensor<float>> layer;

	Tensor<float> in_buf( 4, 6, 0.5f );
	Tensor<float> out_buf( 4, 6 );
	Tensor<float> g_out( 4, 6, 1.0f );
	Tensor<float> g_in( 4, 6 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();
	layer.backward();

	REQUIRE( g_in.rows() == 4u );
	REQUIRE( g_in.cols() == 6u );
}

TEST_CASE( "SoftmaxLayer backward matches Jacobian (two equal logits)", "[softmax_layer][backward][numerical]" )
{
	SoftmaxLayer<Tensor<float>> layer;

	std::vector<float> in_data = { 0.f, 0.f };
	Tensor<float> in_buf( 1, 2, in_data.begin(), in_data.end() );
	Tensor<float> out_buf( 1, 2 );
	std::vector<float> grad_data = { 1.f, 0.f };
	Tensor<float> g_out( 1, 2, grad_data.begin(), grad_data.end() );
	Tensor<float> g_in( 1, 2 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();
	layer.backward();

	REQUIRE_THAT( g_in( 0, 0 ), WithinAbs( 0.25f, eps ) );
	REQUIRE_THAT( g_in( 0, 1 ), WithinAbs( -0.25f, eps ) );
}

TEST_CASE( "SoftmaxLayer backward uniform logits and one-hot upstream", "[softmax_layer][backward][numerical]" )
{
	SoftmaxLayer<Tensor<float>> layer;

	Tensor<float> in_buf( 1, 3, 0.0f );
	Tensor<float> out_buf( 1, 3 );
	std::vector<float> grad_data = { 1.f, 0.f, 0.f };
	Tensor<float> g_out( 1, 3, grad_data.begin(), grad_data.end() );
	Tensor<float> g_in( 1, 3 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();
	layer.backward();

	float const y = 1.0f / 3.0f;
	float const dot = y;
	REQUIRE_THAT( g_in( 0, 0 ), WithinAbs( y * ( 1.f - dot ), eps ) );
	REQUIRE_THAT( g_in( 0, 1 ), WithinAbs( y * ( 0.f - dot ), eps ) );
	REQUIRE_THAT( g_in( 0, 2 ), WithinAbs( y * ( 0.f - dot ), eps ) );
}
