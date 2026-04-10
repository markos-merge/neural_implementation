#include "layer_wiring_helpers.hpp"
#include "linear_layer.hpp"
#include "tensor.hpp"
#include "tensor_like.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

using neural::LinearLayer;
using neural::Tensor;
using neural::test::wire_layer;

static_assert( TensorLike<Tensor<float>> );

using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps = 1e-5f;

} // namespace

TEST_CASE( "LinearLayer forward output shape", "[linear_layer][forward][shape]" )
{
	LinearLayer<Tensor<float>> layer( 5 );

	Tensor<float> in_buf( 2, 3 );
	Tensor<float> out_buf( 2, 5 );
	Tensor<float> g_out( 2, 5 );
	Tensor<float> g_in( 2, 3 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );

	in_buf = Tensor<float>( 2, 3, 1.0f );
	layer.forward();

	REQUIRE( out_buf.rows() == 2u );
	REQUIRE( out_buf.cols() == 5u );
}

TEST_CASE( "LinearLayer forward with known weights", "[linear_layer][forward][numerical]" )
{
	std::vector<float> w_data = { 0.5f, 0.3f, 0.2f, 0.4f };
	std::vector<float> b_data = { 0.1f, 0.2f };
	std::vector<float> x_data = { 1.0f, 2.0f };
	Tensor<float> weights( 2, 2, w_data.begin(), w_data.end() );
	Tensor<float> bias( 2, 1, b_data.begin(), b_data.end() );

	LinearLayer<Tensor<float>> layer( weights, bias );
	Tensor<float> in_buf( 1, 2, x_data.begin(), x_data.end() );
	Tensor<float> out_buf( 1, 2 );
	Tensor<float> g_out( 1, 2 );
	Tensor<float> g_in( 1, 2 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	REQUIRE( out_buf.rows() == 1u );
	REQUIRE( out_buf.cols() == 2u );
	REQUIRE_THAT( out_buf( 0, 0 ), WithinAbs( 1.0f, eps ) );
	REQUIRE_THAT( out_buf( 0, 1 ), WithinAbs( 1.3f, eps ) );
}

TEST_CASE( "LinearLayer backward gradient shapes", "[linear_layer][backward][shape]" )
{
	LinearLayer<Tensor<float>> layer( 5 );

	Tensor<float> in_buf( 2, 3, 1.0f );
	Tensor<float> out_buf( 2, 5 );
	Tensor<float> g_out( 2, 5, 1.0f );
	Tensor<float> g_in( 2, 3 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();

	layer.backward();

	REQUIRE( g_in.rows() == 2u );
	REQUIRE( g_in.cols() == 3u );

	Tensor<float> const &grad_w = layer.getGradWeights();
	Tensor<float> const &grad_b = layer.getGradBias();

	REQUIRE( grad_w.rows() == 3u );
	REQUIRE( grad_w.cols() == 5u );
	REQUIRE( grad_b.rows() == 5u );
	REQUIRE( grad_b.cols() == 1u );
}

TEST_CASE( "LinearLayer backward with known values", "[linear_layer][backward][numerical]" )
{
	std::vector<float> w_data = { 0.6f, 0.7f };
	std::vector<float> b_data = { 0.1f };
	std::vector<float> h_data = { 1.0f, 1.3f };
	Tensor<float> weights( 2, 1, w_data.begin(), w_data.end() );
	Tensor<float> bias( 1, 1, b_data.begin(), b_data.end() );

	LinearLayer<Tensor<float>> layer( weights, bias );
	Tensor<float> in_buf( 1, 2, h_data.begin(), h_data.end() );
	Tensor<float> out_buf( 1, 1 );
	Tensor<float> g_out( 1, 1, -0.39f );
	Tensor<float> g_in( 1, 2 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();
	layer.backward();

	REQUIRE( g_in.rows() == 1u );
	REQUIRE( g_in.cols() == 2u );
	REQUIRE_THAT( g_in( 0, 0 ), WithinAbs( -0.234f, eps ) );
	REQUIRE_THAT( g_in( 0, 1 ), WithinAbs( -0.273f, eps ) );

	Tensor<float> const &grad_w = layer.getGradWeights();
	Tensor<float> const &grad_b = layer.getGradBias();

	REQUIRE_THAT( grad_w( 0, 0 ), WithinAbs( -0.39f, eps ) );
	REQUIRE_THAT( grad_w( 1, 0 ), WithinAbs( -0.507f, eps ) );
	REQUIRE_THAT( grad_b( 0, 0 ), WithinAbs( -0.39f, eps ) );
}

TEST_CASE( "LinearLayer gradient check", "[linear_layer][gradient_check]" )
{
	constexpr float h = 1e-5f;

	std::vector<float> w_data = { 0.5f, 0.3f, 0.2f, 0.4f };
	std::vector<float> b_data = { 0.1f, 0.2f };
	std::vector<float> x_data = { 1.0f, 2.0f };
	Tensor<float> weights( 2, 2, w_data.begin(), w_data.end() );
	Tensor<float> bias( 2, 1, b_data.begin(), b_data.end() );
	Tensor<float> input( 1, 2, x_data.begin(), x_data.end() );

	LinearLayer<Tensor<float>> layer( weights, bias );
	Tensor<float> in_buf( 1, 2 );
	Tensor<float> out_buf( 1, 2 );
	Tensor<float> g_out( 1, 2, 1.0f );
	Tensor<float> g_in( 1, 2 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	in_buf = input;
	layer.forward();
	layer.backward();

	Tensor<float> const &grad_w = layer.getGradWeights();
	Tensor<float> const &grad_b = layer.getGradBias();

	for ( std::size_t i = 0; i < grad_w.rows(); ++i ) {
		for ( std::size_t j = 0; j < grad_w.cols(); ++j ) {
			std::size_t idx = i * grad_w.cols() + j;
			float orig = w_data[idx];

			w_data[idx] = orig + h;
			Tensor<float> w_plus( 2, 2, w_data.begin(), w_data.end() );
			LinearLayer<Tensor<float>> layer_plus( w_plus, bias );
			Tensor<float> in_p( 1, 2 );
			Tensor<float> out_p( 1, 2 );
			Tensor<float> go_p( 1, 2, 1.0f );
			Tensor<float> gi_p( 1, 2 );
			wire_layer( layer_plus, in_p, out_p, go_p, gi_p );
			in_p = input;
			layer_plus.forward();
			float out_plus = out_p.sum();

			w_data[idx] = orig - h;
			Tensor<float> w_minus( 2, 2, w_data.begin(), w_data.end() );
			LinearLayer<Tensor<float>> layer_minus( w_minus, bias );
			Tensor<float> in_m( 1, 2 );
			Tensor<float> out_m( 1, 2 );
			Tensor<float> go_m( 1, 2, 1.0f );
			Tensor<float> gi_m( 1, 2 );
			wire_layer( layer_minus, in_m, out_m, go_m, gi_m );
			in_m = input;
			layer_minus.forward();
			float out_minus = out_m.sum();

			w_data[idx] = orig;

			float num_grad = ( out_plus - out_minus ) / ( 2.0f * h );
			REQUIRE_THAT( grad_w( i, j ), WithinAbs( num_grad, 0.02f ) );
		}
	}

	for ( std::size_t i = 0; i < grad_b.rows(); ++i ) {
		for ( std::size_t j = 0; j < grad_b.cols(); ++j ) {
			std::size_t idx = i * grad_b.cols() + j;
			float orig = b_data[idx];

			b_data[idx] = orig + h;
			Tensor<float> b_plus( 2, 1, b_data.begin(), b_data.end() );
			LinearLayer<Tensor<float>> layer_plus( weights, b_plus );
			Tensor<float> in_p( 1, 2 );
			Tensor<float> out_p( 1, 2 );
			Tensor<float> go_p( 1, 2, 1.0f );
			Tensor<float> gi_p( 1, 2 );
			wire_layer( layer_plus, in_p, out_p, go_p, gi_p );
			in_p = input;
			layer_plus.forward();
			float out_plus = out_p.sum();

			b_data[idx] = orig - h;
			Tensor<float> b_minus( 2, 1, b_data.begin(), b_data.end() );
			LinearLayer<Tensor<float>> layer_minus( weights, b_minus );
			Tensor<float> in_m( 1, 2 );
			Tensor<float> out_m( 1, 2 );
			Tensor<float> go_m( 1, 2, 1.0f );
			Tensor<float> gi_m( 1, 2 );
			wire_layer( layer_minus, in_m, out_m, go_m, gi_m );
			in_m = input;
			layer_minus.forward();
			float out_minus = out_m.sum();

			b_data[idx] = orig;

			float num_grad = ( out_plus - out_minus ) / ( 2.0f * h );
			REQUIRE_THAT( grad_b( i, j ), WithinAbs( num_grad, 0.02f ) );
		}
	}
}
