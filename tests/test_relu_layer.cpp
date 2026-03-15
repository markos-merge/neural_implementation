#include "layer.hpp"
#include "relu_layer.hpp"
#include "tensor.hpp"
#include "tensor_like.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

using neural::ReLULayer;
using neural::Tensor;

static_assert( TensorLike<Tensor<float>> );
static_assert( LayerLike<ReLULayer<Tensor<float>>, Tensor<float>> );

using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps = 1e-5f;

} // namespace

TEST_CASE( "ReLULayer forward output shape", "[relu_layer][forward][shape]" )
{
	ReLULayer<Tensor<float>> layer;

	Tensor<float> input( 2, 5, 1.0f ); // batch=2, features=5
	Tensor<float> output = layer.forward( input );

	REQUIRE( output.rows() == 2u );
	REQUIRE( output.cols() == 5u );
}

TEST_CASE( "ReLULayer forward zeros stay zero", "[relu_layer][forward][numerical]" )
{
	ReLULayer<Tensor<float>> layer;

	std::vector<float> data = { -2.f, -1.f, 0.f, 0.f };
	Tensor<float> input( 2, 2, data.begin(), data.end() );
	Tensor<float> output = layer.forward( input );

	REQUIRE_THAT( output( 0, 0 ), WithinAbs( 0.0f, eps ) );
	REQUIRE_THAT( output( 0, 1 ), WithinAbs( 0.0f, eps ) );
	REQUIRE_THAT( output( 1, 0 ), WithinAbs( 0.0f, eps ) );
	REQUIRE_THAT( output( 1, 1 ), WithinAbs( 0.0f, eps ) );
}

TEST_CASE( "ReLULayer forward positives pass through", "[relu_layer][forward][numerical]" )
{
	ReLULayer<Tensor<float>> layer;

	std::vector<float> data = { 1.f, 2.5f, 3.f, 4.f };
	Tensor<float> input( 2, 2, data.begin(), data.end() );
	Tensor<float> output = layer.forward( input );

	REQUIRE_THAT( output( 0, 0 ), WithinAbs( 1.0f, eps ) );
	REQUIRE_THAT( output( 0, 1 ), WithinAbs( 2.5f, eps ) );
	REQUIRE_THAT( output( 1, 0 ), WithinAbs( 3.0f, eps ) );
	REQUIRE_THAT( output( 1, 1 ), WithinAbs( 4.0f, eps ) );
}

TEST_CASE( "ReLULayer forward mixed positive and negative", "[relu_layer][forward][numerical]" )
{
	ReLULayer<Tensor<float>> layer;

	std::vector<float> data = { -1.f, 1.f, 0.f, 2.5f };
	Tensor<float> input( 2, 2, data.begin(), data.end() );
	Tensor<float> output = layer.forward( input );

	REQUIRE_THAT( output( 0, 0 ), WithinAbs( 0.0f, eps ) );
	REQUIRE_THAT( output( 0, 1 ), WithinAbs( 1.0f, eps ) );
	REQUIRE_THAT( output( 1, 0 ), WithinAbs( 0.0f, eps ) );
	REQUIRE_THAT( output( 1, 1 ), WithinAbs( 2.5f, eps ) );
}

TEST_CASE( "ReLULayer backward gradient shape", "[relu_layer][backward][shape]" )
{
	ReLULayer<Tensor<float>> layer;

	Tensor<float> input( 2, 5, 1.0f );
	layer.forward( input );

	Tensor<float> grad_output( 2, 5, 1.0f );
	Tensor<float> grad_input = layer.backward( grad_output );

	REQUIRE( grad_input.rows() == 2u );
	REQUIRE( grad_input.cols() == 5u );
}

TEST_CASE( "ReLULayer backward gradient flows only where input > 0",
           "[relu_layer][backward][numerical]" )
{
	ReLULayer<Tensor<float>> layer;

	// input: [-1, 1, 0, 2] -> mask: [0, 1, 0, 1]
	std::vector<float> in_data = { -1.f, 1.f, 0.f, 2.f };
	Tensor<float> input( 2, 2, in_data.begin(), in_data.end() );
	layer.forward( input );

	// grad_output: all 1s
	Tensor<float> grad_output( 2, 2, 1.0f );
	Tensor<float> grad_input = layer.backward( grad_output );

	// grad_input = grad_output * mask = [0, 1, 0, 1]
	REQUIRE_THAT( grad_input( 0, 0 ), WithinAbs( 0.0f, eps ) );
	REQUIRE_THAT( grad_input( 0, 1 ), WithinAbs( 1.0f, eps ) );
	REQUIRE_THAT( grad_input( 1, 0 ), WithinAbs( 0.0f, eps ) );
	REQUIRE_THAT( grad_input( 1, 1 ), WithinAbs( 1.0f, eps ) );
}

TEST_CASE( "ReLULayer backward with arbitrary grad_output", "[relu_layer][backward][numerical]" )
{
	ReLULayer<Tensor<float>> layer;

	// input: [1, -1, 2] -> mask: [1, 0, 1]
	std::vector<float> in_data = { 1.f, -1.f, 2.f };
	Tensor<float> input( 1, 3, in_data.begin(), in_data.end() );
	layer.forward( input );

	std::vector<float> grad_data = { 0.5f, 0.3f, 0.8f };
	Tensor<float> grad_output( 1, 3, grad_data.begin(), grad_data.end() );
	Tensor<float> grad_input = layer.backward( grad_output );

	// grad_input = grad_output * mask = [0.5, 0, 0.8]
	REQUIRE_THAT( grad_input( 0, 0 ), WithinAbs( 0.5f, eps ) );
	REQUIRE_THAT( grad_input( 0, 1 ), WithinAbs( 0.0f, eps ) );
	REQUIRE_THAT( grad_input( 0, 2 ), WithinAbs( 0.8f, eps ) );
}
