#include "layer.hpp"
#include "sigmoid_layer.hpp"
#include "tensor.hpp"
#include "tensor_like.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

using neural::SigmoidLayer;
using neural::Tensor;

static_assert( TensorLike<Tensor<float>> );
static_assert( LayerLike<SigmoidLayer<Tensor<float>>, Tensor<float>> );

using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps = 1e-5f;

} // namespace

TEST_CASE( "SigmoidLayer forward output shape", "[sigmoid_layer][forward][shape]" )
{
	SigmoidLayer<Tensor<float>> layer;

	Tensor<float> input( 2, 5, 1.0f );
	Tensor<float> output = layer.forward( input );

	REQUIRE( output.rows() == 2u );
	REQUIRE( output.cols() == 5u );
}

TEST_CASE( "SigmoidLayer forward output in range (0, 1)", "[sigmoid_layer][forward][numerical]" )
{
	SigmoidLayer<Tensor<float>> layer;

	std::vector<float> data = { -10.f, 0.f, 10.f };
	Tensor<float> input( 1, 3, data.begin(), data.end() );
	Tensor<float> output = layer.forward( input );

	// σ(-10) ≈ 0, σ(0) = 0.5, σ(10) ≈ 1
	REQUIRE( output( 0, 0 ) > 0.0f );
	REQUIRE( output( 0, 0 ) < 0.001f );
	REQUIRE_THAT( output( 0, 1 ), WithinAbs( 0.5f, eps ) );
	REQUIRE( output( 0, 2 ) > 0.999f );
	REQUIRE( output( 0, 2 ) < 1.001f );
}

TEST_CASE( "SigmoidLayer forward sigmoid(0) equals 0.5", "[sigmoid_layer][forward][numerical]" )
{
	SigmoidLayer<Tensor<float>> layer;

	Tensor<float> input( 2, 2, 0.0f );
	Tensor<float> output = layer.forward( input );

	for ( std::size_t i = 0; i < output.rows(); ++i )
		for ( std::size_t j = 0; j < output.cols(); ++j )
			REQUIRE_THAT( output( i, j ), WithinAbs( 0.5f, eps ) );
}

TEST_CASE( "SigmoidLayer backward gradient shape", "[sigmoid_layer][backward][shape]" )
{
	SigmoidLayer<Tensor<float>> layer;

	Tensor<float> input( 2, 5, 1.0f );
	layer.forward( input );

	Tensor<float> grad_output( 2, 5, 1.0f );
	Tensor<float> grad_input = layer.backward( grad_output );

	REQUIRE( grad_input.rows() == 2u );
	REQUIRE( grad_input.cols() == 5u );
}

TEST_CASE( "SigmoidLayer backward with known values", "[sigmoid_layer][backward][numerical]" )
{
	// ∂L/∂x = (∂L/∂y) · y · (1 - y)
	// At x=0: y=0.5, (1-y)=0.5, so if ∂L/∂y=1: ∂L/∂x = 0.25
	SigmoidLayer<Tensor<float>> layer;

	Tensor<float> input( 1, 1, 0.0f );
	layer.forward( input );

	Tensor<float> grad_output( 1, 1, 1.0f );
	Tensor<float> grad_input = layer.backward( grad_output );

	REQUIRE_THAT( grad_input( 0, 0 ), WithinAbs( 0.25f, eps ) );
}
