#include "cross_entropy_softmax_loss.hpp"
#include "cuda_tensor.hpp"
#include "layer_wiring_helpers.hpp"
#include "linear_layer.hpp"
#include "mse_loss.hpp"
#include "tensor.hpp"
#include "neural_cuda_runtime.hpp"
#include "relu_layer.hpp"
#include "sequential_nn_static.hpp"
#include "sgd_optimizer.hpp"
#include "tensor_like.hpp" // TensorLike (global concept)
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

using neural::CudaTensor;
using neural::LinearLayer;
using neural::MSELoss;
using neural::ReLULayer;
using neural::SequentialNN_static;
using neural::SGDOptimizer;
using neural::SoftmaxCrossEntropyLoss;
using neural::Tensor;
using neural::test::wire_layer;
using Catch::Matchers::WithinAbs;

static_assert( TensorLike<CudaTensor<float>> );

namespace {

constexpr float eps = 1e-4f;

LinearLayer<CudaTensor<float>> make_identity_linear_cuda( std::size_t dim )
{
	std::vector<float> w_data( dim * dim, 0.0f );
	for ( std::size_t i = 0; i < dim; ++i ) {
		w_data[i * dim + i] = 1.0f;
	}
	std::vector<float> b_data( dim, 0.0f );
	CudaTensor<float> weights( dim, dim, w_data.begin(), w_data.end() );
	CudaTensor<float> bias( dim, 1, b_data.begin(), b_data.end() );
	return LinearLayer<CudaTensor<float>>( weights, bias );
}

} // namespace

TEST_CASE( "CudaTensor is TensorLike (compile-time)", "[cuda][nn][tensor_like]" )
{
	REQUIRE( true );
}

TEST_CASE( "LinearLayer CudaTensor forward output shape", "[cuda][linear_layer][shape]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	LinearLayer<CudaTensor<float>> layer( 5 );
	CudaTensor<float> in_buf( 2, 3, 1.0f );
	CudaTensor<float> out_buf( 2, 5 );
	CudaTensor<float> g_out( 2, 5 );
	CudaTensor<float> g_in( 2, 3 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();
	REQUIRE( out_buf.rows() == 2u );
	REQUIRE( out_buf.cols() == 5u );
}

TEST_CASE( "LinearLayer CudaTensor forward matches CPU (1x1)", "[cuda][linear_layer][numerical]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	// Single-feature layer: y = 2*x + 0.5 — exercises matmul + bias without ambiguous GEMM layout.
	CudaTensor<float> weights( 1, 1, 2.f );
	CudaTensor<float> bias( 1, 1, 0.5f );
	CudaTensor<float> input( 1, 1, 3.f );

	LinearLayer<CudaTensor<float>> layer( weights, bias );
	CudaTensor<float> out_buf( 1, 1 );
	CudaTensor<float> g_out( 1, 1 );
	CudaTensor<float> g_in( 1, 1 );
	wire_layer( layer, input, out_buf, g_out, g_in );
	layer.forward();

	Tensor<float> w_cpu( 1, 1, 2.f );
	Tensor<float> b_cpu( 1, 1, 0.5f );
	Tensor<float> x_cpu( 1, 1, 3.f );
	LinearLayer<Tensor<float>> layer_cpu( w_cpu, b_cpu );
	Tensor<float> out_cpu_buf( 1, 1 );
	Tensor<float> g_out_cpu( 1, 1 );
	Tensor<float> g_in_cpu( 1, 1 );
	wire_layer( layer_cpu, x_cpu, out_cpu_buf, g_out_cpu, g_in_cpu );
	layer_cpu.forward();

	REQUIRE( out_buf.rows() == 1u );
	REQUIRE( out_buf.cols() == 1u );
	REQUIRE_THAT( out_buf( 0, 0 ), WithinAbs( out_cpu_buf( 0, 0 ), eps ) );
}

TEST_CASE( "LinearLayer CudaTensor forward matches CPU (2x2 matmul)", "[cuda][linear_layer][numerical]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::vector<float> w_data = { 0.5f, 0.3f, 0.2f, 0.4f };
	std::vector<float> b_data = { 0.1f, 0.2f };
	std::vector<float> x_data = { 1.0f, 2.0f };
	CudaTensor<float> weights( 2, 2, w_data.begin(), w_data.end() );
	CudaTensor<float> bias( 2, 1, b_data.begin(), b_data.end() );
	CudaTensor<float> input( 1, 2, x_data.begin(), x_data.end() );

	LinearLayer<CudaTensor<float>> layer( weights, bias );
	CudaTensor<float> out_buf( 1, 2 );
	CudaTensor<float> g_out( 1, 2 );
	CudaTensor<float> g_in( 1, 2 );
	wire_layer( layer, input, out_buf, g_out, g_in );
	layer.forward();

	Tensor<float> w_cpu( 2, 2, w_data.begin(), w_data.end() );
	Tensor<float> b_cpu( 2, 1, b_data.begin(), b_data.end() );
	Tensor<float> x_cpu( 1, 2, x_data.begin(), x_data.end() );
	LinearLayer<Tensor<float>> layer_cpu( w_cpu, b_cpu );
	Tensor<float> out_cpu_buf( 1, 2 );
	Tensor<float> g_out_cpu( 1, 2 );
	Tensor<float> g_in_cpu( 1, 2 );
	wire_layer( layer_cpu, x_cpu, out_cpu_buf, g_out_cpu, g_in_cpu );
	layer_cpu.forward();

	REQUIRE_THAT( out_buf( 0, 0 ), WithinAbs( out_cpu_buf( 0, 0 ), eps ) );
	REQUIRE_THAT( out_buf( 0, 1 ), WithinAbs( out_cpu_buf( 0, 1 ), eps ) );
}

TEST_CASE( "LinearLayer CudaTensor backward gradient shapes", "[cuda][linear_layer][backward]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	LinearLayer<CudaTensor<float>> layer( 5 );
	CudaTensor<float> in_buf( 2, 3, 1.0f );
	CudaTensor<float> out_buf( 2, 5 );
	CudaTensor<float> g_out( 2, 5, 1.0f );
	CudaTensor<float> g_in( 2, 3 );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );
	layer.forward();
	layer.backward();
	REQUIRE( g_in.rows() == 2u );
	REQUIRE( g_in.cols() == 3u );
	REQUIRE( layer.getGradWeights().rows() == 3u );
	REQUIRE( layer.getGradWeights().cols() == 5u );
	REQUIRE( layer.getGradBias().rows() == 5u );
	REQUIRE( layer.getGradBias().cols() == 1u );
}

TEST_CASE( "SequentialNN CudaTensor trainStep MSE zero for identity", "[cuda][sequential_nn][trainStep]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	auto linear = make_identity_linear_cuda( 2 );
	SequentialNN_static<CudaTensor<float>, MSELoss<CudaTensor<float>>, LinearLayer<CudaTensor<float>>> nn(
	    linear );

	std::vector<float> x_data = { 1.0f, 2.0f };
	std::vector<float> t_data = { 1.0f, 2.0f };
	CudaTensor<float> input( 1, 2, x_data.begin(), x_data.end() );
	CudaTensor<float> target( 1, 2, t_data.begin(), t_data.end() );

	auto const loss = nn.trainStep( input, target );
	REQUIRE_THAT( loss, WithinAbs( 0.0f, eps ) );
}

TEST_CASE( "SequentialNN CudaTensor trainStep softmax CE matches standalone", "[cuda][sequential_nn][softmax_ce]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	auto linear = make_identity_linear_cuda( 3 );
	SequentialNN_static<CudaTensor<float>, SoftmaxCrossEntropyLoss<CudaTensor<float>>,
	             LinearLayer<CudaTensor<float>>>
	    nn( linear );

	std::vector<float> x_data = { 1.0f, 0.0f, 0.0f };
	std::vector<float> t_data = { 1.0f, 0.0f, 0.0f };
	CudaTensor<float> input( 1, 3, x_data.begin(), x_data.end() );
	CudaTensor<float> target( 1, 3, t_data.begin(), t_data.end() );

	auto const loss = nn.trainStep( input, target );
	CudaTensor<float> logits = nn.forward( input );
	SoftmaxCrossEntropyLoss<CudaTensor<float>> standalone;
	auto const expected = standalone.forward( logits, target )( 0, 0 );
	REQUIRE_THAT( loss, WithinAbs( expected, eps ) );
}

TEST_CASE( "SGDOptimizer CudaTensor train runs", "[cuda][sgd_optimizer]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	using SimpleNN =
	    SequentialNN_static<CudaTensor<float>, MSELoss<CudaTensor<float>>, LinearLayer<CudaTensor<float>>>;
	auto linear = LinearLayer<CudaTensor<float>>( 1 );
	SimpleNN nn( linear );
	SGDOptimizer<CudaTensor<float>, SimpleNN> opt( nn );

	std::vector<Tensor<float>> inputs;
	std::vector<Tensor<float>> targets;
	for ( int i = 0; i < 4; ++i ) {
		inputs.push_back( Tensor<float>( 1, 2, static_cast<float>( i ) / 4.f ) );
		targets.push_back( Tensor<float>( 1, 1, static_cast<float>( i ) / 8.f ) );
	}
	opt.m_epochs = 2;
	opt.m_batch_size = 2;
	REQUIRE_NOTHROW( opt.train( inputs, targets ) );
}

TEST_CASE( "SGDOptimizer CudaTensor assignTensorAsRow batching", "[cuda][sgd_optimizer][batch]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	using SimpleNN =
	    SequentialNN_static<CudaTensor<float>, MSELoss<CudaTensor<float>>, LinearLayer<CudaTensor<float>>>;
	auto linear = LinearLayer<CudaTensor<float>>( 1 );
	SimpleNN nn( linear );

	std::vector<CudaTensor<float>> inputs;
	std::vector<CudaTensor<float>> targets;
	inputs.push_back( CudaTensor<float>( 1, 2, 1.0f ) );
	targets.push_back( CudaTensor<float>( 1, 1, 2.0f ) );
	inputs.push_back( CudaTensor<float>( 1, 2, 0.5f ) );
	targets.push_back( CudaTensor<float>( 1, 1, 1.0f ) );

	CudaTensor<float> batch_in( 2, 2, 0.f );
	CudaTensor<float> batch_t( 2, 1, 0.f );
	std::vector<std::size_t> idx = { 1u, 0u };
	for ( std::size_t r = 0; r < 2u; ++r ) {
		batch_in.assignTensorAsRow( r, inputs[idx[r]] );
		batch_t.assignTensorAsRow( r, targets[idx[r]] );
	}
	auto const loss = nn.trainStep( batch_in, batch_t );
	REQUIRE( loss >= 0.0f );
}
