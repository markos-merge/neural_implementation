#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <vector>

#if NEURAL_CUDA_ENABLED && NEURAL_CUDNN_ENABLED

#include "convolutional_box_static.hpp"
#include "convolutional_layer.hpp"
#include "cross_entropy_softmax_loss.hpp"
#include "cuda_tensor.hpp"
#include "cuda_tensor_n.hpp"
#include "dropout_layer.hpp"
#include "linear_layer.hpp"
#include "max_pool_layer.hpp"
#include "metrics.hpp"
#include "momentum_sgd_optimizer.hpp"
#include "neural_cuda_runtime.hpp"
#include "relu_layer.hpp"
#include "sequential_nn_static.hpp"
#include "tensor.hpp"

using Catch::Matchers::WithinAbs;

namespace {

using neural::ConvolutionalBox_static;
using neural::ConvolutionalLayer;
using neural::CudaTensor;
using neural::CudaTensor4;
using neural::DropoutLayer;
using neural::LinearLayer;
using neural::MaxPoolLayer;
using neural::MomentumSGDOptimizer;
using neural::ReLULayer;
using neural::SequentialNN_static;
using neural::SoftmaxCrossEntropyLoss;
using neural::Tensor;

using Tensor2D_t = CudaTensor<float>;
using TensorN_t  = CudaTensor4<float>;

constexpr int C = 2;
constexpr int H = 8;
constexpr int W = 8;
constexpr std::size_t in_cols = static_cast<std::size_t>( C * H * W );
constexpr std::size_t n_classes = 3;

using ConvFeatureExtractor = ConvolutionalBox_static<
    TensorN_t, Tensor2D_t,
    ConvolutionalLayer<TensorN_t>,
    ReLULayer<TensorN_t>,
    DropoutLayer<TensorN_t>,
    MaxPoolLayer<TensorN_t>>;

/// Full model is a \c SequentialNN (2D latch tensor): conv stack + linear head.
using TinyConvNN_t =
    SequentialNN_static<Tensor2D_t, SoftmaxCrossEntropyLoss<Tensor2D_t>, ConvFeatureExtractor,
                 LinearLayer<Tensor2D_t>>;

void set_train_mode( TinyConvNN_t &nn, bool training )
{
	nn.forEachLayer( [training]( auto &layer ) {
		if constexpr ( requires { layer.setTraining( training ); } ) {
			layer.setTraining( training );
		}
	} );
}

/// NCHW row-major flat index for batch row layout expected by \c ConvolutionalBox.
void fill_spatial_pattern( Tensor<float> &row, int cls )
{
	for ( int c = 0; c < C; ++c ) {
		for ( int y = 0; y < H; ++y ) {
			for ( int x = 0; x < W; ++x ) {
				std::size_t const idx =
				    static_cast<std::size_t>( c * H * W + y * W + x );
				float v = 0.1f;
				if ( cls == 0 ) {
					v = ( x < W / 2 ) ? 1.f : 0.15f;
				} else if ( cls == 1 ) {
					v = ( x >= W / 2 ) ? 1.f : 0.15f;
				} else {
					v = ( ( ( x + y + c ) & 1 ) == 0 ) ? 0.95f : 0.12f;
				}
				row.assign( 0, idx, v );
			}
		}
	}
}

} // namespace

TEST_CASE( "Tiny CUDA conv net: momentum train lowers loss; eval dropout is deterministic",
           "[cuda][conv_net][momentum][dropout]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}

	TinyConvNN_t nn(
	    ConvFeatureExtractor(
	        static_cast<std::size_t>( C ), static_cast<std::size_t>( H ), static_cast<std::size_t>( W ),
	        ConvolutionalLayer<TensorN_t>( 8, 3 ),
	        ReLULayer<TensorN_t>(),
	        DropoutLayer<TensorN_t>( 1.0f, 424242u ),
	        MaxPoolLayer<TensorN_t>( 2, 2 ) ),
	    LinearLayer<Tensor2D_t>( n_classes ) );

	MomentumSGDOptimizer<Tensor2D_t, TinyConvNN_t> opt( nn );
	opt.m_learning_rate  = 0.04f;
	opt.m_momentum       = 0.9f;
	opt.m_batch_size     = 12;
	opt.m_epochs         = 18;
	opt.m_l2_regularizer = 1e-4f;
	opt.m_nesterov       = true;

	std::size_t const n_per_class = 16;
	std::vector<Tensor<float>> inputs;
	std::vector<Tensor<float>> targets;
	inputs.reserve( n_per_class * n_classes );
	targets.reserve( n_per_class * n_classes );
	for ( std::size_t rep = 0; rep < n_per_class; ++rep ) {
		for ( std::size_t cls = 0; cls < n_classes; ++cls ) {
			Tensor<float> x( 1, in_cols, 0.f );
			fill_spatial_pattern( x, static_cast<int>( cls ) );
			inputs.push_back( std::move( x ) );
			Tensor<float> t( 1, n_classes, 0.f );
			t.assign( 0, cls, 1.f );
			targets.push_back( std::move( t ) );
		}
	}

	set_train_mode( nn, true );

	float sum_loss_epoch0     = 0.f;
	float sum_loss_last_epoch = 0.f;
	std::size_t n_batches0    = 0;
	std::size_t n_batches_last = 0;

	opt.train(
	    inputs, targets,
	    [&]( std::size_t epoch, std::size_t epoch_max, std::size_t batch_idx,
	         std::size_t batch_max, float loss ) -> bool {
		    (void)epoch_max;
		    (void)batch_idx;
		    (void)batch_max;
		    if ( epoch == 0 ) {
			    sum_loss_epoch0 += loss;
			    ++n_batches0;
		    }
		    if ( epoch + 1 == opt.m_epochs ) {
			    sum_loss_last_epoch += loss;
			    ++n_batches_last;
		    }
		    return false;
	    } );

	REQUIRE( n_batches0 > 0 );
	REQUIRE( n_batches_last > 0 );
	float const mean0 = sum_loss_epoch0 / static_cast<float>( n_batches0 );
	float const mean1 = sum_loss_last_epoch / static_cast<float>( n_batches_last );
	REQUIRE( mean0 > 0.f );
	REQUIRE( mean1 > 0.f );
	REQUIRE( mean1 < mean0 * 0.92f );

	set_train_mode( nn, false );
	float const acc = neural::compute_accuracy( nn, inputs, targets, n_classes, opt.m_batch_size );
	REQUIRE( acc >= 0.55f );

	// Eval: repeated forwards match (dropout inactive).
	nn.ensureBuffersForShape( 4, in_cols );
	Tensor<float> host_batch( 4, in_cols );
	for ( std::size_t r = 0; r < 4; ++r ) {
		host_batch.assignTensorAsRow( r, inputs[r] );
	}
	CudaTensor<float> batch_in( 4, in_cols );
	batch_in.assign( host_batch.data(), host_batch.size() );
	CudaTensor<float> out1 = nn.forward( batch_in );
	CudaTensor<float> out2 = nn.forward( batch_in );
	for ( std::size_t r = 0; r < 4; ++r ) {
		for ( std::size_t c = 0; c < n_classes; ++c ) {
			REQUIRE_THAT( out1( r, c ), WithinAbs( out2( r, c ), 1e-5f ) );
		}
	}
}

#endif // NEURAL_CUDA_ENABLED && NEURAL_CUDNN_ENABLED
