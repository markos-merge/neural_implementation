#include "layer_properties.hpp"
#include "layer_wiring_helpers.hpp"
#include "relu_layer.hpp"
#include "tensor.hpp"
#include "tensor_like.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

using neural::ReLULayer;
using neural::Tensor;
using neural::test::wire_layer;

static_assert( TensorLike<Tensor<float>> );
static_assert( LayerLike<ReLULayer<Tensor<float>>, Tensor<float>> );

using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps = 1e-5f;

} // namespace

TEST_CASE( "ReLULayer forward output shape", "[relu_layer][forward][shape]" )
{
	ReLULayer<Tensor<float>> layer;
	Tensor<float> in_buf( 2, 5, 1.0f );
	Tensor<float> out_buf( 2, 5 );
	Tensor<float> g_out( 2, 5 );
	Tensor<float> g_in( 2, 5 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	REQUIRE( out_buf.rows() == 2u );
	REQUIRE( out_buf.cols() == 5u );
}

TEST_CASE( "ReLULayer forward zeros stay zero", "[relu_layer][forward][numerical]" )
{
	ReLULayer<Tensor<float>> layer;

	std::vector<float> data = { -2.f, -1.f, 0.f, 0.f };
	Tensor<float> in_buf( 2, 2, data.begin(), data.end() );
	Tensor<float> out_buf( 2, 2 );
	Tensor<float> g_out( 2, 2 );
	Tensor<float> g_in( 2, 2 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	REQUIRE_THAT( out_buf( 0, 0 ), WithinAbs( 0.0f, eps ) );
	REQUIRE_THAT( out_buf( 0, 1 ), WithinAbs( 0.0f, eps ) );
	REQUIRE_THAT( out_buf( 1, 0 ), WithinAbs( 0.0f, eps ) );
	REQUIRE_THAT( out_buf( 1, 1 ), WithinAbs( 0.0f, eps ) );
}

TEST_CASE( "ReLULayer forward positives pass through", "[relu_layer][forward][numerical]" )
{
	ReLULayer<Tensor<float>> layer;

	std::vector<float> data = { 1.f, 2.5f, 3.f, 4.f };
	Tensor<float> in_buf( 2, 2, data.begin(), data.end() );
	Tensor<float> out_buf( 2, 2 );
	Tensor<float> g_out( 2, 2 );
	Tensor<float> g_in( 2, 2 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	REQUIRE_THAT( out_buf( 0, 0 ), WithinAbs( 1.0f, eps ) );
	REQUIRE_THAT( out_buf( 0, 1 ), WithinAbs( 2.5f, eps ) );
	REQUIRE_THAT( out_buf( 1, 0 ), WithinAbs( 3.0f, eps ) );
	REQUIRE_THAT( out_buf( 1, 1 ), WithinAbs( 4.0f, eps ) );
}

TEST_CASE( "ReLULayer forward mixed positive and negative", "[relu_layer][forward][numerical]" )
{
	ReLULayer<Tensor<float>> layer;

	std::vector<float> data = { -1.f, 1.f, 0.f, 2.5f };
	Tensor<float> in_buf( 2, 2, data.begin(), data.end() );
	Tensor<float> out_buf( 2, 2 );
	Tensor<float> g_out( 2, 2 );
	Tensor<float> g_in( 2, 2 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	REQUIRE_THAT( out_buf( 0, 0 ), WithinAbs( 0.0f, eps ) );
	REQUIRE_THAT( out_buf( 0, 1 ), WithinAbs( 1.0f, eps ) );
	REQUIRE_THAT( out_buf( 1, 0 ), WithinAbs( 0.0f, eps ) );
	REQUIRE_THAT( out_buf( 1, 1 ), WithinAbs( 2.5f, eps ) );
}

TEST_CASE( "ReLULayer backward gradient shape", "[relu_layer][backward][shape]" )
{
	ReLULayer<Tensor<float>> layer;

	Tensor<float> in_buf( 2, 5, 1.0f );
	Tensor<float> out_buf( 2, 5 );
	Tensor<float> g_out( 2, 5, 1.0f );
	Tensor<float> g_in( 2, 5 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();
	layer.backward();

	REQUIRE( g_in.rows() == 2u );
	REQUIRE( g_in.cols() == 5u );
}

TEST_CASE( "ReLULayer backward gradient flows only where input > 0",
           "[relu_layer][backward][numerical]" )
{
	ReLULayer<Tensor<float>> layer;

	std::vector<float> in_data = { -1.f, 1.f, 0.f, 2.f };
	Tensor<float> in_buf( 2, 2, in_data.begin(), in_data.end() );
	Tensor<float> out_buf( 2, 2 );
	Tensor<float> g_out( 2, 2, 1.0f );
	Tensor<float> g_in( 2, 2 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();
	layer.backward();

	REQUIRE_THAT( g_in( 0, 0 ), WithinAbs( 0.0f, eps ) );
	REQUIRE_THAT( g_in( 0, 1 ), WithinAbs( 1.0f, eps ) );
	REQUIRE_THAT( g_in( 1, 0 ), WithinAbs( 0.0f, eps ) );
	REQUIRE_THAT( g_in( 1, 1 ), WithinAbs( 1.0f, eps ) );
}

TEST_CASE( "ReLULayer backward with arbitrary grad_output", "[relu_layer][backward][numerical]" )
{
	ReLULayer<Tensor<float>> layer;

	std::vector<float> in_data = { 1.f, -1.f, 2.f };
	Tensor<float> in_buf( 1, 3, in_data.begin(), in_data.end() );
	Tensor<float> out_buf( 1, 3 );
	std::vector<float> grad_data = { 0.5f, 0.3f, 0.8f };
	Tensor<float> g_out( 1, 3, grad_data.begin(), grad_data.end() );
	Tensor<float> g_in( 1, 3 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();
	layer.backward();

	REQUIRE_THAT( g_in( 0, 0 ), WithinAbs( 0.5f, eps ) );
	REQUIRE_THAT( g_in( 0, 1 ), WithinAbs( 0.0f, eps ) );
	REQUIRE_THAT( g_in( 0, 2 ), WithinAbs( 0.8f, eps ) );
}
