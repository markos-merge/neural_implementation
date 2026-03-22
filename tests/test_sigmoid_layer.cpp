#include "layer.hpp"
#include "layer_wiring_helpers.hpp"
#include "sigmoid_layer.hpp"
#include "tensor.hpp"
#include "tensor_like.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

using neural::SigmoidLayer;
using neural::Tensor;
using neural::test::wire_layer;

static_assert( TensorLike<Tensor<float>> );
static_assert( LayerLike<SigmoidLayer<Tensor<float>>, Tensor<float>> );

using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps = 1e-5f;

} // namespace

TEST_CASE( "SigmoidLayer forward output shape", "[sigmoid_layer][forward][shape]" )
{
	SigmoidLayer<Tensor<float>> layer;
	Tensor<float> in_buf( 2, 5, 1.0f );
	Tensor<float> out_buf( 2, 5 );
	Tensor<float> g_out( 2, 5 );
	Tensor<float> g_in( 2, 5 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	REQUIRE( out_buf.rows() == 2u );
	REQUIRE( out_buf.cols() == 5u );
}

TEST_CASE( "SigmoidLayer forward output in range (0, 1)", "[sigmoid_layer][forward][numerical]" )
{
	SigmoidLayer<Tensor<float>> layer;

	std::vector<float> data = { -10.f, 0.f, 10.f };
	Tensor<float> in_buf( 1, 3, data.begin(), data.end() );
	Tensor<float> out_buf( 1, 3 );
	Tensor<float> g_out( 1, 3 );
	Tensor<float> g_in( 1, 3 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	REQUIRE( out_buf( 0, 0 ) > 0.0f );
	REQUIRE( out_buf( 0, 0 ) < 0.001f );
	REQUIRE_THAT( out_buf( 0, 1 ), WithinAbs( 0.5f, eps ) );
	REQUIRE( out_buf( 0, 2 ) > 0.999f );
	REQUIRE( out_buf( 0, 2 ) < 1.001f );
}

TEST_CASE( "SigmoidLayer forward sigmoid(0) equals 0.5", "[sigmoid_layer][forward][numerical]" )
{
	SigmoidLayer<Tensor<float>> layer;

	Tensor<float> in_buf( 2, 2, 0.0f );
	Tensor<float> out_buf( 2, 2 );
	Tensor<float> g_out( 2, 2 );
	Tensor<float> g_in( 2, 2 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	for ( std::size_t i = 0; i < out_buf.rows(); ++i )
		for ( std::size_t j = 0; j < out_buf.cols(); ++j )
			REQUIRE_THAT( out_buf( i, j ), WithinAbs( 0.5f, eps ) );
}

TEST_CASE( "SigmoidLayer backward gradient shape", "[sigmoid_layer][backward][shape]" )
{
	SigmoidLayer<Tensor<float>> layer;

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

TEST_CASE( "SigmoidLayer backward with known values", "[sigmoid_layer][backward][numerical]" )
{
	SigmoidLayer<Tensor<float>> layer;

	Tensor<float> in_buf( 1, 1, 0.0f );
	Tensor<float> out_buf( 1, 1 );
	Tensor<float> g_out( 1, 1, 1.0f );
	Tensor<float> g_in( 1, 1 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();
	layer.backward();

	REQUIRE_THAT( g_in( 0, 0 ), WithinAbs( 0.25f, eps ) );
}
