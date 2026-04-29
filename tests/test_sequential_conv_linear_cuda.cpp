#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>

#if NEURAL_CUDA_ENABLED && NEURAL_CUDNN_ENABLED

#include "convolutional_box.hpp"
#include "convolutional_layer.hpp"
#include "cross_entropy_softmax_loss.hpp"
#include "cuda_tensor.hpp"
#include "cuda_tensor_n.hpp"
#include "linear_layer.hpp"
#include "max_pool_layer.hpp"
#include "neural_cuda_runtime.hpp"
#include "relu_layer.hpp"
#include "sequential_nn.hpp"
#include "tensor.hpp"

namespace {

using neural::ConvolutionalBox;
using neural::ConvolutionalLayer;
using neural::CudaTensor;
using neural::CudaTensor4;
using neural::LinearLayer;
using neural::MaxPoolLayer;
using neural::ReLULayer;
using neural::SequentialNN;
using neural::SoftmaxCrossEntropyLoss;
using neural::Tensor;

using Tensor2D_t = CudaTensor<float>;
using TensorN_t  = CudaTensor4<float>;

// Tiny RGB-ish patch so conv + pool produce a valid flat vector for the linear head.
constexpr int C = 2;
constexpr int H = 8;
constexpr int W = 8;
constexpr std::size_t in_cols  = static_cast<std::size_t>( C * H * W );
constexpr std::size_t n_logits = 3;

using SmallBox = ConvolutionalBox<
    TensorN_t, Tensor2D_t,
    ConvolutionalLayer<TensorN_t>, ReLULayer<TensorN_t>, MaxPoolLayer<TensorN_t>
>;

using SmallConvLinearNN = SequentialNN<
    Tensor2D_t, SoftmaxCrossEntropyLoss<Tensor2D_t>, SmallBox, LinearLayer<Tensor2D_t>>;

} // namespace

/// Smoke test: \c SequentialNN with \c ConvolutionalBox then \c LinearLayer on device (forward + one train step).
TEST_CASE( "SequentialNN CUDA: conv box + linear forward and trainStep", "[cuda][sequential_nn][conv][linear]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}

	SmallBox box(
	    static_cast<std::size_t>( C ), static_cast<std::size_t>( H ), static_cast<std::size_t>( W ),
	    ConvolutionalLayer<TensorN_t>( 8, 3 ),
	    ReLULayer<TensorN_t>(),
	    MaxPoolLayer<TensorN_t>( 2, 2 ) );

	SmallConvLinearNN nn( box, LinearLayer<Tensor2D_t>( n_logits ) );

	nn.ensureBuffersForShape( 1, in_cols );

	Tensor<float> h_in( 1, in_cols, 0.1f );
	Tensor<float> h_t( 1, n_logits, 0.f );
	h_t.assign( 0, 1, 1.f );
	Tensor2D_t d_in( 1, in_cols );
	d_in.assign( h_in.data(), h_in.size() );
	Tensor2D_t d_t( 1, n_logits );
	d_t.assign( h_t.data(), h_t.size() );

	Tensor2D_t logits = nn.forward( d_in );
	REQUIRE( logits.rows() == 1u );
	REQUIRE( logits.cols() == n_logits );

	float const loss = nn.trainStep( d_in, d_t );
	REQUIRE( std::isfinite( loss ) );
	REQUIRE( loss > 0.f );

	nn.ensureBuffersForShape( 2, in_cols );
	Tensor<float> h_in2( 2, in_cols, 0.12f );
	Tensor<float> h_t2( 2, n_logits, 0.f );
	h_t2.assign( 0, 0, 1.f );
	h_t2.assign( 1, 2, 1.f );
	Tensor2D_t d2_in( 2, in_cols );
	d2_in.assign( h_in2.data(), h_in2.size() );
	Tensor2D_t d2_t( 2, n_logits );
	d2_t.assign( h_t2.data(), h_t2.size() );
	float const loss2 = nn.trainStep( d2_in, d2_t );
	REQUIRE( std::isfinite( loss2 ) );
}

#endif // NEURAL_CUDA_ENABLED && NEURAL_CUDNN_ENABLED
