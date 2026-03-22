#include "layer.hpp"
#include "softmax_layer.hpp"
#include "tensor.hpp"
#include "tensor_like.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

using neural::SoftmaxLayer;
using neural::Tensor;

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

	Tensor<float> input( 3, 7, 1.0f );
	Tensor<float> output = layer.forward( input );

	REQUIRE( output.rows() == 3u );
	REQUIRE( output.cols() == 7u );
}

TEST_CASE( "SoftmaxLayer forward rows sum to one", "[softmax_layer][forward][numerical]" )
{
	SoftmaxLayer<Tensor<float>> layer;

	std::vector<float> data = { 0.f, 1.f, -0.5f, 2.f, 0.f, 0.25f, -1.f, 3.f };
	Tensor<float> input( 2, 4, data.begin(), data.end() );
	Tensor<float> output = layer.forward( input );

	for ( std::size_t i = 0; i < output.rows(); ++i ) {
		float row_sum = 0.f;
		for ( std::size_t j = 0; j < output.cols(); ++j )
			row_sum += output( i, j );
		REQUIRE_THAT( row_sum, WithinAbs( 1.0f, eps ) );
	}
}

TEST_CASE( "SoftmaxLayer forward values match reference (single row)", "[softmax_layer][forward][numerical]" )
{
	SoftmaxLayer<Tensor<float>> layer;

	std::vector<float> logits = { 1.f, 2.f, 3.f };
	Tensor<float> input( 1, 3, logits.begin(), logits.end() );
	Tensor<float> output = layer.forward( input );

	for ( std::size_t j = 0; j < 3u; ++j )
		REQUIRE_THAT( output( 0, j ), WithinAbs( softmax_ref_row( logits, j ), eps ) );
}

TEST_CASE( "SoftmaxLayer forward equal logits are uniform", "[softmax_layer][forward][numerical]" )
{
	SoftmaxLayer<Tensor<float>> layer;

	Tensor<float> input( 2, 4, 0.0f );
	Tensor<float> output = layer.forward( input );

	float const u = 1.0f / 4.0f;
	for ( std::size_t i = 0; i < output.rows(); ++i )
		for ( std::size_t j = 0; j < output.cols(); ++j )
			REQUIRE_THAT( output( i, j ), WithinAbs( u, eps ) );
}

TEST_CASE( "SoftmaxLayer forward one dominant logit", "[softmax_layer][forward][numerical]" )
{
	SoftmaxLayer<Tensor<float>> layer;

	std::vector<float> data = { -5.f, 5.f, -5.f };
	Tensor<float> input( 1, 3, data.begin(), data.end() );
	Tensor<float> output = layer.forward( input );

	REQUIRE( output( 0, 1 ) > 0.99f );
	REQUIRE( output( 0, 0 ) < 0.01f );
	REQUIRE( output( 0, 2 ) < 0.01f );
}

TEST_CASE( "SoftmaxLayer backward gradient shape", "[softmax_layer][backward][shape]" )
{
	SoftmaxLayer<Tensor<float>> layer;

	Tensor<float> input( 4, 6, 0.5f );
	layer.forward( input );

	Tensor<float> grad_output( 4, 6, 1.0f );
	Tensor<float> grad_input = layer.backward( grad_output );

	REQUIRE( grad_input.rows() == 4u );
	REQUIRE( grad_input.cols() == 6u );
}

TEST_CASE( "SoftmaxLayer backward matches Jacobian (two equal logits)", "[softmax_layer][backward][numerical]" )
{
	// y = [0.5, 0.5], grad_y = [1, 0]  =>  grad_x = [0.25, -0.25]
	SoftmaxLayer<Tensor<float>> layer;

	std::vector<float> in_data = { 0.f, 0.f };
	Tensor<float> input( 1, 2, in_data.begin(), in_data.end() );
	layer.forward( input );

	std::vector<float> grad_data = { 1.f, 0.f };
	Tensor<float> grad_output( 1, 2, grad_data.begin(), grad_data.end() );
	Tensor<float> grad_input = layer.backward( grad_output );

	REQUIRE_THAT( grad_input( 0, 0 ), WithinAbs( 0.25f, eps ) );
	REQUIRE_THAT( grad_input( 0, 1 ), WithinAbs( -0.25f, eps ) );
}

TEST_CASE( "SoftmaxLayer backward uniform logits and one-hot upstream", "[softmax_layer][backward][numerical]" )
{
	SoftmaxLayer<Tensor<float>> layer;

	Tensor<float> input( 1, 3, 0.0f );
	layer.forward( input );

	std::vector<float> grad_data = { 1.f, 0.f, 0.f };
	Tensor<float> grad_output( 1, 3, grad_data.begin(), grad_data.end() );
	Tensor<float> grad_input = layer.backward( grad_output );

	float const y = 1.0f / 3.0f;
	float const dot = y;
	REQUIRE_THAT( grad_input( 0, 0 ), WithinAbs( y * ( 1.f - dot ), eps ) );
	REQUIRE_THAT( grad_input( 0, 1 ), WithinAbs( y * ( 0.f - dot ), eps ) );
	REQUIRE_THAT( grad_input( 0, 2 ), WithinAbs( y * ( 0.f - dot ), eps ) );
}
