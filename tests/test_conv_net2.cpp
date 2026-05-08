#include "refactor/sequential_nn.hpp"
#include "refactor/sgd_optimizer.hpp"
#include "refactor/momentum_sgd_optimizer.hpp"
#include "refactor/cross_entropy_loss.hpp"
#include "refactor/mse_loss.hpp"
#include "refactor/convolutional_layer.hpp"
#include "refactor/linear_layer.hpp"
#include "refactor/relu_layer.hpp"
#include "refactor/max_pool_layer.hpp"
#include "refactor/batch_norm_layer.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <array>
#include <cmath>
#include <vector>

using neural::refactor::BatchNormLayer;
using neural::refactor::Cpu;
using neural::refactor::ConvolutionalLayer;
using neural::refactor::CrossEntropyLoss;
using neural::refactor::LinearLayer;
using neural::refactor::MaxPoolLayer;
using neural::refactor::MomentumSGDOptimizer;
using neural::refactor::ReLULayer;
using neural::refactor::SGDOptimizer;
using neural::refactor::SequentialNN2;
using Catch::Matchers::WithinAbs;

namespace {

bool all_finite( float const *p, std::size_t n )
{
	for ( std::size_t i = 0; i < n; ++i )
		if ( !std::isfinite( p[i] ) )
			return false;
	return true;
}

} // namespace

// =============================================================================
// Conv(entry) + ReLU + Linear: forward produces finite output of correct shape
// =============================================================================

TEST_CASE( "Conv+ReLU+Linear forward: finite output and correct shape",
           "[conv_net2][cpu][forward]" )
{
	// Input: N=2 images, C=1 H=8 W=8
	constexpr std::size_t N = 2, C = 1, H = 8, W = 8;
	constexpr std::size_t in_features = C * H * W;
	constexpr std::size_t out_classes = 4;

	SequentialNN2<float, Cpu> nn;
	nn.addLayer(
	    ConvolutionalLayer<float, Cpu>( 4, 3, std::array<std::size_t, 3>{ C, H, W }, 0, 0 ) );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( LinearLayer<float, Cpu>( out_classes ) );
	nn.wire();

	std::vector<float> data( N * in_features );
	for ( std::size_t i = 0; i < data.size(); ++i )
		data[i] = static_cast<float>( i % 7 ) - 3.f;

	nn.forward( data.data(), N, in_features );

	auto const shape = nn.outputShape();
	REQUIRE( shape[0] == N );
	REQUIRE( shape[1] == out_classes );
	REQUIRE( all_finite( nn.output(), N * out_classes ) );
}

// =============================================================================
// Conv + ReLU + MaxPool + Linear: forward + backward runs without crash
// =============================================================================

TEST_CASE( "Conv+ReLU+MaxPool+Linear forward+backward runs without crash",
           "[conv_net2][cpu][backward]" )
{
	constexpr std::size_t N = 2, C = 1, H = 8, W = 8;
	constexpr std::size_t in_features = C * H * W;
	constexpr std::size_t out_classes = 3;

	SequentialNN2<float, Cpu> nn;
	nn.addLayer(
	    ConvolutionalLayer<float, Cpu>( 4, 3, std::array<std::size_t, 3>{ C, H, W }, 0, 0 ) );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( MaxPoolLayer<float, Cpu>( 2, 2 ) );
	nn.addLayer( LinearLayer<float, Cpu>( out_classes ) );
	nn.wire();

	std::vector<float> data( N * in_features );
	for ( std::size_t i = 0; i < data.size(); ++i )
		data[i] = static_cast<float>( i % 5 ) - 2.f;

	nn.forward( data.data(), N, in_features );
	REQUIRE( all_finite( nn.output(), nn.outputSize() ) );

	// Simple one-hot grad
	std::vector<float> grad( nn.outputSize(), 0.f );
	for ( std::size_t n = 0; n < N; ++n )
		grad[n * out_classes + ( n % out_classes )] = 1.f;

	REQUIRE_NOTHROW( nn.backward( grad.data() ) );
}

// =============================================================================
// SGD step on Conv+Linear: output stays finite
// =============================================================================

TEST_CASE( "SGDOptimizer step on Conv+Linear: output stays finite",
           "[conv_net2][cpu][sgd][finite]" )
{
	constexpr std::size_t N = 2, C = 1, H = 6, W = 6;
	constexpr std::size_t in_features = C * H * W;
	constexpr std::size_t out_classes = 3;

	SequentialNN2<float, Cpu> nn;
	nn.addLayer(
	    ConvolutionalLayer<float, Cpu>( 4, 3, std::array<std::size_t, 3>{ C, H, W }, 0, 0 ) );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( LinearLayer<float, Cpu>( out_classes ) );
	nn.wire();

	SGDOptimizer<float, Cpu> opt( 0.01f, CrossEntropyLoss<float, Cpu>{} );

	std::vector<float> data( N * in_features );
	for ( std::size_t i = 0; i < data.size(); ++i )
		data[i] = static_cast<float>( i % 7 ) - 3.f;

	// one-hot targets
	std::vector<float> targets( N * out_classes, 0.f );
	targets[0 * out_classes + 0] = 1.f;
	targets[1 * out_classes + 1] = 1.f;

	REQUIRE_NOTHROW( opt.step( nn, data.data(), N, in_features, targets.data() ) );
	REQUIRE( all_finite( nn.output(), nn.outputSize() ) );
}

// =============================================================================
// Conv + BN + ReLU + MaxPool + Linear: SGD 10 steps stay finite
// =============================================================================

TEST_CASE( "Conv+BN+ReLU+MaxPool+Linear SGD 10 steps: stays finite",
           "[conv_net2][cpu][sgd][batchnorm]" )
{
	constexpr std::size_t N = 4, C = 1, H = 8, W = 8;
	constexpr std::size_t in_features = C * H * W;
	constexpr std::size_t out_classes = 4;

	SequentialNN2<float, Cpu> nn;
	nn.addLayer(
	    ConvolutionalLayer<float, Cpu>( 8, 3, std::array<std::size_t, 3>{ C, H, W }, 0, 0 ) );
	nn.addLayer( BatchNormLayer<float, Cpu>() );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( MaxPoolLayer<float, Cpu>( 2, 2 ) );
	nn.addLayer( LinearLayer<float, Cpu>( out_classes ) );
	nn.wire();

	SGDOptimizer<float, Cpu> opt( 0.01f, CrossEntropyLoss<float, Cpu>{} );

	std::vector<float> data( N * in_features );
	for ( std::size_t i = 0; i < data.size(); ++i )
		data[i] = ( static_cast<float>( i % 13 ) - 6.f ) * 0.1f;

	std::vector<float> targets( N * out_classes, 0.f );
	for ( std::size_t n = 0; n < N; ++n )
		targets[n * out_classes + ( n % out_classes )] = 1.f;

	for ( int i = 0; i < 10; ++i )
		REQUIRE_NOTHROW( opt.step( nn, data.data(), N, in_features, targets.data() ) );

	REQUIRE( all_finite( nn.output(), nn.outputSize() ) );
}

// =============================================================================
// MomentumSGD on Conv+ReLU+MaxPool+Linear: 10 steps stay finite
// =============================================================================

TEST_CASE( "MomentumSGD on Conv+ReLU+MaxPool+Linear: 10 steps stay finite",
           "[conv_net2][cpu][momentum][finite]" )
{
	constexpr std::size_t N = 3, C = 1, H = 8, W = 8;
	constexpr std::size_t in_features = C * H * W;
	constexpr std::size_t out_classes = 3;

	SequentialNN2<float, Cpu> nn;
	nn.addLayer(
	    ConvolutionalLayer<float, Cpu>( 4, 3, std::array<std::size_t, 3>{ C, H, W }, 0, 0 ) );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( MaxPoolLayer<float, Cpu>( 2, 2 ) );
	nn.addLayer( LinearLayer<float, Cpu>( out_classes ) );
	nn.wire();

	MomentumSGDOptimizer<float, Cpu> opt( 0.01f, 0.9f, CrossEntropyLoss<float, Cpu>{} );

	std::vector<float> data( N * in_features );
	for ( std::size_t i = 0; i < data.size(); ++i )
		data[i] = ( static_cast<float>( i % 9 ) - 4.f ) * 0.2f;

	std::vector<float> targets( N * out_classes, 0.f );
	for ( std::size_t n = 0; n < N; ++n )
		targets[n * out_classes + ( n % out_classes )] = 1.f;

	for ( int i = 0; i < 10; ++i )
		REQUIRE_NOTHROW( opt.step( nn, data.data(), N, in_features, targets.data() ) );

	REQUIRE( all_finite( nn.output(), nn.outputSize() ) );
}

// =============================================================================
// Two-stage conv net: Conv+ReLU+Pool × 2 + Linear — forward shape is correct
// =============================================================================

TEST_CASE( "Two-stage conv net: output shape is correct", "[conv_net2][cpu][multistage][shape]" )
{
	// 1×10×10 input → Conv(4,3): 4×8×8 → Pool(2,2): 4×4×4 → Conv(8,3): 8×2×2 → Pool(2,2): 8×1×1
	// → Linear(5): N×5
	constexpr std::size_t N = 2, C = 1, H = 10, W = 10;
	constexpr std::size_t in_features = C * H * W;
	constexpr std::size_t out_f       = 5;

	SequentialNN2<float, Cpu> nn;
	nn.addLayer(
	    ConvolutionalLayer<float, Cpu>( 4, 3, std::array<std::size_t, 3>{ C, H, W }, 0, 0 ) );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( MaxPoolLayer<float, Cpu>( 2, 2 ) );
	nn.addLayer( ConvolutionalLayer<float, Cpu>( 8, 3, 0, 0 ) );
	nn.addLayer( ReLULayer<float, Cpu>() );
	nn.addLayer( MaxPoolLayer<float, Cpu>( 2, 2 ) );
	nn.addLayer( LinearLayer<float, Cpu>( out_f ) );
	nn.wire();

	std::vector<float> data( N * in_features );
	for ( std::size_t i = 0; i < data.size(); ++i )
		data[i] = static_cast<float>( i % 11 ) * 0.1f;

	nn.forward( data.data(), N, in_features );

	auto const shape = nn.outputShape();
	REQUIRE( shape[0] == N );
	REQUIRE( shape[1] == out_f );
	REQUIRE( all_finite( nn.output(), nn.outputSize() ) );
}
