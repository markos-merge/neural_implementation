#include "cross_entropy_softmax_loss.hpp"
#include "linear_layer.hpp"
#include "mse_loss.hpp"
#include "relu_layer.hpp"
#include "sequential_nn.hpp"
#include "sigmoid_layer.hpp"
#include "tensor.hpp"
#include "tensor_like.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

using neural::LinearLayer;
using neural::MSELoss;
using neural::ReLULayer;
using neural::SequentialNN;
using neural::SoftmaxCrossEntropyLoss;
using neural::SigmoidLayer;
using neural::Tensor;

static_assert( TensorLike<Tensor<float>> );

using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps = 1e-5f;

// Identity linear layer: y = x (weights = I, bias = 0)
LinearLayer<Tensor<float>> make_identity_linear( std::size_t dim )
{
	std::vector<float> w_data( dim * dim, 0.0f );
	for ( std::size_t i = 0; i < dim; ++i )
		w_data[i * dim + i] = 1.0f;
	std::vector<float> b_data( dim, 0.0f );
	Tensor<float> weights( dim, dim, w_data.begin(), w_data.end() );
	Tensor<float> bias( dim, 1, b_data.begin(), b_data.end() );
	return LinearLayer<Tensor<float>>( weights, bias );
}

} // namespace

TEST_CASE( "SequentialNN single layer forward output shape", "[sequential_nn][forward][shape]" )
{
	auto linear = LinearLayer<Tensor<float>>( 3, 5 );
	SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>> nn( linear );

	Tensor<float> input( 2, 3, 1.0f );
	Tensor<float> output = nn.forward( input );

	REQUIRE( output.rows() == 2u );
	REQUIRE( output.cols() == 5u );
}

TEST_CASE( "SequentialNN multiple layers forward output shape", "[sequential_nn][forward][shape]" )
{
	auto linear1 = LinearLayer<Tensor<float>>( 3, 4 );
	auto relu = ReLULayer<Tensor<float>>();
	auto linear2 = LinearLayer<Tensor<float>>( 4, 2 );
	SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>, ReLULayer<Tensor<float>>,
	             LinearLayer<Tensor<float>>>
	    nn( linear1, relu, linear2 );

	Tensor<float> input( 2, 3, 1.0f );
	Tensor<float> output = nn.forward( input );

	REQUIRE( output.rows() == 2u );
	REQUIRE( output.cols() == 2u );
}

TEST_CASE( "SequentialNN single identity layer forward preserves input", "[sequential_nn][forward][numerical]" )
{
	auto linear = make_identity_linear( 2 );
	SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>> nn( linear );

	std::vector<float> x_data = { 1.0f, 2.0f };
	Tensor<float> input( 1, 2, x_data.begin(), x_data.end() );
	Tensor<float> output = nn.forward( input );

	REQUIRE_THAT( output( 0, 0 ), WithinAbs( 1.0f, eps ) );
	REQUIRE_THAT( output( 0, 1 ), WithinAbs( 2.0f, eps ) );
}

TEST_CASE( "SequentialNN two identity layers forward preserves input", "[sequential_nn][forward][numerical]" )
{
	auto linear1 = make_identity_linear( 2 );
	auto linear2 = make_identity_linear( 2 );
	SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>, LinearLayer<Tensor<float>>>
	    nn( linear1, linear2 );

	std::vector<float> x_data = { 1.0f, 2.0f };
	Tensor<float> input( 1, 2, x_data.begin(), x_data.end() );
	Tensor<float> output = nn.forward( input );

	REQUIRE_THAT( output( 0, 0 ), WithinAbs( 1.0f, eps ) );
	REQUIRE_THAT( output( 0, 1 ), WithinAbs( 2.0f, eps ) );
}

TEST_CASE( "SequentialNN backward gradient shape", "[sequential_nn][backward][shape]" )
{
	auto linear = LinearLayer<Tensor<float>>( 3, 5 );
	SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>> nn( linear );

	Tensor<float> input( 2, 3, 1.0f );
	nn.forward( input );

	Tensor<float> grad_output( 2, 5, 1.0f );
	Tensor<float> grad_input = nn.backward( grad_output );

	REQUIRE( grad_input.rows() == 2u );
	REQUIRE( grad_input.cols() == 3u );
}

TEST_CASE( "SequentialNN backward through multiple layers gradient shape", "[sequential_nn][backward][shape]" )
{
	auto linear1 = LinearLayer<Tensor<float>>( 3, 4 );
	auto relu = ReLULayer<Tensor<float>>();
	auto linear2 = LinearLayer<Tensor<float>>( 4, 2 );
	SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>, ReLULayer<Tensor<float>>,
	             LinearLayer<Tensor<float>>>
	    nn( linear1, relu, linear2 );

	Tensor<float> input( 2, 3, 1.0f );
	nn.forward( input );

	Tensor<float> grad_output( 2, 2, 1.0f );
	Tensor<float> grad_input = nn.backward( grad_output );

	REQUIRE( grad_input.rows() == 2u );
	REQUIRE( grad_input.cols() == 3u );
}

TEST_CASE( "SequentialNN trainStep returns loss", "[sequential_nn][trainStep]" )
{
	auto linear = make_identity_linear( 2 );
	SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>> nn( linear );

	std::vector<float> x_data = { 1.0f, 2.0f };
	std::vector<float> t_data = { 1.0f, 2.0f };
	Tensor<float> input( 1, 2, x_data.begin(), x_data.end() );
	Tensor<float> target( 1, 2, t_data.begin(), t_data.end() );

	auto loss = nn.trainStep( input, target );

	REQUIRE_THAT( loss, WithinAbs( 0.0f, eps ) );
}

TEST_CASE( "SequentialNN trainStep loss when pred equals target", "[sequential_nn][trainStep][numerical]" )
{
	auto linear = make_identity_linear( 2 );
	SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>> nn( linear );

	std::vector<float> data = { 3.0f, 4.0f };
	Tensor<float> input( 1, 2, data.begin(), data.end() );
	Tensor<float> target( 1, 2, data.begin(), data.end() );

	auto loss = nn.trainStep( input, target );

	REQUIRE_THAT( loss, WithinAbs( 0.0f, eps ) );
}

TEST_CASE( "SequentialNN trainStep loss when pred differs from target", "[sequential_nn][trainStep][numerical]" )
{
	auto linear = make_identity_linear( 2 );
	SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>> nn( linear );

	// pred = [1, 2], target = [3, 4] -> MSE = ((1-3)^2 + (2-4)^2)/2 = (4+4)/2 = 4
	std::vector<float> pred_data = { 1.0f, 2.0f };
	std::vector<float> target_data = { 3.0f, 4.0f };
	Tensor<float> input( 1, 2, pred_data.begin(), pred_data.end() );
	Tensor<float> target( 1, 2, target_data.begin(), target_data.end() );

	auto loss = nn.trainStep( input, target );

	REQUIRE_THAT( loss, WithinAbs( 4.0f, eps ) );
}

TEST_CASE( "SequentialNN trainStep with Linear-ReLU-Linear pipeline", "[sequential_nn][trainStep]" )
{
	// Known weights: x -> linear -> relu -> linear -> y
	// Use simple weights so we can verify loss is computed
	std::vector<float> w1_data = { 1.0f, 0.0f, 0.0f, 1.0f };
	std::vector<float> b1_data = { 0.0f, 0.0f };
	std::vector<float> w2_data = { 1.0f, 0.0f, 0.0f, 1.0f };
	std::vector<float> b2_data = { 0.0f, 0.0f };
	Tensor<float> weights1( 2, 2, w1_data.begin(), w1_data.end() );
	Tensor<float> bias1( 2, 1, b1_data.begin(), b1_data.end() );
	Tensor<float> weights2( 2, 2, w2_data.begin(), w2_data.end() );
	Tensor<float> bias2( 2, 1, b2_data.begin(), b2_data.end() );

	LinearLayer<Tensor<float>> linear1( weights1, bias1 );
	ReLULayer<Tensor<float>> relu;
	LinearLayer<Tensor<float>> linear2( weights2, bias2 );

	SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>, ReLULayer<Tensor<float>>,
	             LinearLayer<Tensor<float>>>
	    nn( linear1, relu, linear2 );

	// input [1, 2] -> linear1 -> [1, 2] -> relu -> [1, 2] -> linear2 -> [1, 2]
	// target [1, 2] -> loss = 0
	std::vector<float> x_data = { 1.0f, 2.0f };
	std::vector<float> t_data = { 1.0f, 2.0f };
	Tensor<float> input( 1, 2, x_data.begin(), x_data.end() );
	Tensor<float> target( 1, 2, t_data.begin(), t_data.end() );

	auto loss = nn.trainStep( input, target );

	REQUIRE_THAT( loss, WithinAbs( 0.0f, eps ) );
}

TEST_CASE( "SequentialNN forward then backward preserves gradient flow", "[sequential_nn][backward][numerical]" )
{
	// Identity pipeline: grad_output should equal grad_input
	auto linear = make_identity_linear( 2 );
	SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>> nn( linear );

	std::vector<float> x_data = { 1.0f, 2.0f };
	std::vector<float> grad_data = { 0.5f, -0.3f };
	Tensor<float> input( 1, 2, x_data.begin(), x_data.end() );
	nn.forward( input );

	Tensor<float> grad_output( 1, 2, grad_data.begin(), grad_data.end() );
	Tensor<float> grad_input = nn.backward( grad_output );

	REQUIRE_THAT( grad_input( 0, 0 ), WithinAbs( 0.5f, eps ) );
	REQUIRE_THAT( grad_input( 0, 1 ), WithinAbs( -0.3f, eps ) );
}

TEST_CASE( "SequentialNN trainStep softmax cross-entropy matches standalone loss", "[sequential_nn][trainStep][softmax_ce]" )
{
	auto linear = make_identity_linear( 3 );
	SequentialNN<Tensor<float>, SoftmaxCrossEntropyLoss<Tensor<float>>, LinearLayer<Tensor<float>>>
	    nn( linear );

	std::vector<float> x_data = { 1.0f, 0.0f, 0.0f };
	std::vector<float> t_data = { 1.0f, 0.0f, 0.0f };
	Tensor<float> input( 1, 3, x_data.begin(), x_data.end() );
	Tensor<float> target( 1, 3, t_data.begin(), t_data.end() );

	auto const loss = nn.trainStep( input, target );

	Tensor<float> logits = nn.forward( input );
	SoftmaxCrossEntropyLoss<Tensor<float>> standalone;
	auto const expected = standalone.forward( logits, target )( 0, 0 );

	REQUIRE_THAT( loss, WithinAbs( expected, eps ) );
}
