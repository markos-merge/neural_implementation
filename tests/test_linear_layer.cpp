#include "linear_layer.hpp"
#include "tensor.hpp"
#include "tensor_like.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

using neural::LinearLayer;
using neural::Tensor;

static_assert( TensorLike<Tensor<float>> );

using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps = 1e-5f;

} // namespace

TEST_CASE( "LinearLayer forward output shape", "[linear_layer][forward][shape]" )
{
	LinearLayer<Tensor<float>> layer( 3, 5 );

	Tensor<float> input( 2, 3, 1.0f ); // batch=2, in_features=3
	Tensor<float> output = layer.forward( input );

	REQUIRE( output.rows() == 2u );
	REQUIRE( output.cols() == 5u );
}

TEST_CASE( "LinearLayer forward with known weights", "[linear_layer][forward][numerical]" )
{
	// From BACKPROP_WALKTHROUGH: x=[1,2], W₁=[0.5 0.3; 0.2 0.4], b₁=[0.1; 0.2]
	// Expected h = [1.0, 1.3]
	std::vector<float> w_data = { 0.5f, 0.3f, 0.2f, 0.4f };
	std::vector<float> b_data = { 0.1f, 0.2f };
	std::vector<float> x_data = { 1.0f, 2.0f };
	Tensor<float> weights( 2, 2, w_data.begin(), w_data.end() );
	Tensor<float> bias( 2, 1, b_data.begin(), b_data.end() );
	Tensor<float> input( 1, 2, x_data.begin(), x_data.end() );

	LinearLayer<Tensor<float>> layer( weights, bias );
	Tensor<float> output = layer.forward( input );

	REQUIRE( output.rows() == 1u );
	REQUIRE( output.cols() == 2u );
	REQUIRE_THAT( output( 0, 0 ), WithinAbs( 1.0f, eps ) );
	REQUIRE_THAT( output( 0, 1 ), WithinAbs( 1.3f, eps ) );
}

TEST_CASE( "LinearLayer backward gradient shapes", "[linear_layer][backward][shape]" )
{
	LinearLayer<Tensor<float>> layer( 3, 5 );

	Tensor<float> input( 2, 3, 1.0f );
	layer.forward( input );

	Tensor<float> grad_output( 2, 5, 1.0f );
	Tensor<float> grad_input = layer.backward( grad_output );

	REQUIRE( grad_input.rows() == 2u );
	REQUIRE( grad_input.cols() == 3u );

	Tensor<float> const &grad_w = layer.getGradWeights();
	Tensor<float> const &grad_b = layer.getGradBias();

	REQUIRE( grad_w.rows() == 3u );
	REQUIRE( grad_w.cols() == 5u );
	REQUIRE( grad_b.rows() == 5u );
	REQUIRE( grad_b.cols() == 1u );
}

TEST_CASE( "LinearLayer backward with known values", "[linear_layer][backward][numerical]" )
{
	// Linear₂ from BACKPROP_WALKTHROUGH: h=[1,1.3], W₂=[0.6; 0.7], b₂=[0.1]
	// Forward: ŷ = [1.61], ∂L/∂ŷ = [-0.39]
	// Backward: ∂L/∂h = [-0.234, -0.273], ∂L/∂W₂ = [-0.39; -0.507], ∂L/∂b₂ = [-0.39]
	std::vector<float> w_data = { 0.6f, 0.7f };
	std::vector<float> b_data = { 0.1f };
	std::vector<float> h_data = { 1.0f, 1.3f };
	Tensor<float> weights( 2, 1, w_data.begin(), w_data.end() );
	Tensor<float> bias( 1, 1, b_data.begin(), b_data.end() );
	Tensor<float> input( 1, 2, h_data.begin(), h_data.end() );

	LinearLayer<Tensor<float>> layer( weights, bias );
	layer.forward( input );

	Tensor<float> grad_output( 1, 1, -0.39f );
	Tensor<float> grad_input = layer.backward( grad_output );

	REQUIRE( grad_input.rows() == 1u );
	REQUIRE( grad_input.cols() == 2u );
	REQUIRE_THAT( grad_input( 0, 0 ), WithinAbs( -0.234f, eps ) );
	REQUIRE_THAT( grad_input( 0, 1 ), WithinAbs( -0.273f, eps ) );

	Tensor<float> const &grad_w = layer.getGradWeights();
	Tensor<float> const &grad_b = layer.getGradBias();

	REQUIRE_THAT( grad_w( 0, 0 ), WithinAbs( -0.39f, eps ) );
	REQUIRE_THAT( grad_w( 1, 0 ), WithinAbs( -0.507f, eps ) );
	REQUIRE_THAT( grad_b( 0, 0 ), WithinAbs( -0.39f, eps ) );
}

TEST_CASE( "LinearLayer gradient check", "[linear_layer][gradient_check]" )
{
	// Compare analytical gradients to numerical gradients (finite differences)
	constexpr float h = 1e-5f;

	std::vector<float> w_data = { 0.5f, 0.3f, 0.2f, 0.4f };
	std::vector<float> b_data = { 0.1f, 0.2f };
	std::vector<float> x_data = { 1.0f, 2.0f };
	Tensor<float> weights( 2, 2, w_data.begin(), w_data.end() );
	Tensor<float> bias( 2, 1, b_data.begin(), b_data.end() );
	Tensor<float> input( 1, 2, x_data.begin(), x_data.end() );

	LinearLayer<Tensor<float>> layer( weights, bias );
	layer.forward( input );

	// Simple loss: sum of output (L = sum(ŷ)), so dL/dy = 1 for each element
	Tensor<float> grad_output( 1, 2, 1.0f );
	layer.backward( grad_output );

	Tensor<float> const &grad_w = layer.getGradWeights();
	Tensor<float> const &grad_b = layer.getGradBias();

	// Numerical gradient for weights: (L(w+ε) - L(w-ε)) / (2ε)
	for ( std::size_t i = 0; i < grad_w.rows(); ++i ) {
		for ( std::size_t j = 0; j < grad_w.cols(); ++j ) {
			std::size_t idx = i * grad_w.cols() + j;
			float orig = w_data[idx];

			w_data[idx] = orig + h;
			Tensor<float> w_plus( 2, 2, w_data.begin(), w_data.end() );
			LinearLayer<Tensor<float>> layer_plus( w_plus, bias );
			float out_plus = layer_plus.forward( input ).sum();

			w_data[idx] = orig - h;
			Tensor<float> w_minus( 2, 2, w_data.begin(), w_data.end() );
			LinearLayer<Tensor<float>> layer_minus( w_minus, bias );
			float out_minus = layer_minus.forward( input ).sum();

			w_data[idx] = orig;

			float num_grad = ( out_plus - out_minus ) / ( 2.0f * h );
			REQUIRE_THAT( grad_w( i, j ), WithinAbs( num_grad, 0.02f ) );
		}
	}

	// Numerical gradient for bias
	for ( std::size_t i = 0; i < grad_b.rows(); ++i ) {
		for ( std::size_t j = 0; j < grad_b.cols(); ++j ) {
			std::size_t idx = i * grad_b.cols() + j;
			float orig = b_data[idx];

			b_data[idx] = orig + h;
			Tensor<float> b_plus( 2, 1, b_data.begin(), b_data.end() );
			LinearLayer<Tensor<float>> layer_plus( weights, b_plus );
			float out_plus = layer_plus.forward( input ).sum();

			b_data[idx] = orig - h;
			Tensor<float> b_minus( 2, 1, b_data.begin(), b_data.end() );
			LinearLayer<Tensor<float>> layer_minus( weights, b_minus );
			float out_minus = layer_minus.forward( input ).sum();

			b_data[idx] = orig;

			float num_grad = ( out_plus - out_minus ) / ( 2.0f * h );
			REQUIRE_THAT( grad_b( i, j ), WithinAbs( num_grad, 0.02f ) );
		}
	}
}
