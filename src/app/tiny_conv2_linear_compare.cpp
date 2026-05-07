#include "tiny_conv2_linear_compare.hpp"
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if !NEURAL_CUDA_ENABLED || !NEURAL_CUDNN_ENABLED

void run_tiny_conv2_linear_compare( std::filesystem::path const & )
{
	std::cout << "tiny_conv2_linear_compare: build with -DNEURAL_ENABLE_CUDA=ON (cuDNN required).\n";
}

int main( int argc, char **argv )
{
	(void)argc;
	(void)argv;
	run_tiny_conv2_linear_compare( {} );
	return 1;
}

#else

#include "convolutional_box_static.hpp"
#include "convolutional_layer.hpp"
#include "cross_entropy_softmax_loss.hpp"
#include "cuda_tensor.hpp"
#include "cuda_tensor_n.hpp"
#include "linear_layer.hpp"
#include "neural_cuda_layer_sync.hpp"
#include "neural_cuda_runtime.hpp"
#include "relu_layer.hpp"
#include "sgd_optimizer.hpp"
#include "sequential_nn_static.hpp"
#include "tensor.hpp"
#include <cuda_runtime_api.h>

namespace {

using Tensor2D_t = neural::CudaTensor<float>;
using TensorN_t  = neural::CudaTensor4<float>;

// Keep in sync with scripts/tiny_conv2_linear_compare_pytorch.py
constexpr int C = 3;
constexpr int H = 8;
constexpr int W = 8;
constexpr std::size_t in_cols = static_cast<std::size_t>( C * H * W );
constexpr std::size_t conv1_out = 4;
constexpr std::size_t conv2_out = 2;
constexpr std::size_t linear_out = 5;
constexpr std::uint64_t k_global_seed = 42u;

constexpr std::size_t conv1_w_elems = conv1_out * 3u * 3u * 3u;
constexpr std::size_t conv2_w_elems = conv2_out * conv1_out * 3u * 3u;
constexpr std::size_t flat_elems    = conv2_out * static_cast<std::size_t>( H ) * static_cast<std::size_t>( W );
constexpr std::size_t linear_w_elems = flat_elems * linear_out;

/// Categorical target (one-hot column index). Sync with Python \c TARGET_CLASS.
constexpr std::size_t target_class = 2u;
/// SGD step — sync with Python \c LR / \c WEIGHT_DECAY.
constexpr float k_lr            = 0.01f;
constexpr float k_weight_decay = 0.0f;

void write_f32_file( std::filesystem::path const &p, std::vector<float> const &v )
{
	std::ofstream f( p, std::ios::binary );
	if ( !f ) {
		throw std::runtime_error( "tiny_conv2_linear_compare: cannot write " + p.string() );
	}
	f.write( reinterpret_cast<char const *>( v.data() ),
	         static_cast<std::streamsize>( v.size() * sizeof( float ) ) );
}

void append_he( std::vector<float> &buf, std::size_t n, std::size_t fan_in, std::mt19937_64 &gen )
{
	double const scale = std::sqrt( 2.0 / static_cast<double>( fan_in ) );
	std::uniform_real_distribution<double> dis( -scale, scale );
	std::size_t const old = buf.size();
	buf.resize( old + n );
	for ( std::size_t i = 0; i < n; ++i ) {
		buf[old + i] = static_cast<float>( dis( gen ) );
	}
}

using TinyBox = neural::ConvolutionalBox_static<
    TensorN_t, Tensor2D_t,
    neural::ConvolutionalLayer<TensorN_t>, neural::ReLULayer<TensorN_t>,
    neural::ConvolutionalLayer<TensorN_t>, neural::ReLULayer<TensorN_t>>;

using TinyNN = neural::SequentialNN_static<
    Tensor2D_t, neural::SoftmaxCrossEntropyLoss<Tensor2D_t>, TinyBox,
    neural::LinearLayer<Tensor2D_t>>;

void apply_host_weights( TinyNN &nn, std::vector<float> const &conv1_w, std::vector<float> const &conv2_w,
                         std::vector<float> const &lin_w )
{
	nn.forEachLayer( [&]( auto &lyr ) {
		if constexpr ( requires { lyr.forEachLayer( []( auto & ) {} ); } ) {
			int conv_idx = 0;
			lyr.forEachLayer( [&]( auto &inner ) {
				if constexpr ( requires { inner.getWeights(); inner.kernelSize(); } ) {
					auto &w = inner.getWeights();
					if ( conv_idx == 0 ) {
						if ( conv1_w.size() != w.size() ) {
							throw std::logic_error( "conv1 weight size mismatch" );
						}
						w.assign( conv1_w );
					} else {
						if ( conv2_w.size() != w.size() ) {
							throw std::logic_error( "conv2 weight size mismatch" );
						}
						w.assign( conv2_w );
					}
					++conv_idx;
				}
			} );
		} else if constexpr ( requires { lyr.getWeights(); lyr.outFeatures(); } ) {
			auto &w = lyr.getWeights();
			if ( lin_w.size() != w.size() ) {
				throw std::logic_error( "linear weight size mismatch" );
			}
			w.assign( lin_w.data(), lin_w.size() );
		}
	} );
}

void cuda4_to_file( TensorN_t const &d, std::filesystem::path const &p )
{
	std::vector<float> host( d.size() );
	if ( cudaMemcpy( host.data(), d.data(), d.size() * sizeof( float ), cudaMemcpyDeviceToHost )
	     != cudaSuccess ) {
		throw std::runtime_error( "tiny_conv2_linear_compare: cudaMemcpy conv tensor D2H failed" );
	}
	write_f32_file( p, host );
}

void cuda2_to_file( Tensor2D_t const &d, std::filesystem::path const &p )
{
	std::vector<float> host( d.size() );
	d.copyToHost( host.data() );
	write_f32_file( p, host );
}

void dump_gradients( TinyNN &nn, std::filesystem::path const &artifact_dir )
{
	nn.forEachLayer( [&]( auto &lyr ) {
		if constexpr ( requires { lyr.forEachLayer( []( auto & ) {} ); } ) {
			int conv_idx = 0;
			lyr.forEachLayer( [&]( auto &inner ) {
				if constexpr ( requires { inner.getGradWeights(); inner.kernelSize(); } ) {
					if ( conv_idx == 0 ) {
						cuda4_to_file( inner.getGradWeights(), artifact_dir / "grad_conv1_w.f32" );
						cuda4_to_file( inner.getGradBias(), artifact_dir / "grad_conv1_b.f32" );
					} else {
						cuda4_to_file( inner.getGradWeights(), artifact_dir / "grad_conv2_w.f32" );
						cuda4_to_file( inner.getGradBias(), artifact_dir / "grad_conv2_b.f32" );
					}
					++conv_idx;
				}
			} );
		} else if constexpr ( requires { lyr.getGradWeights(); lyr.outFeatures(); } ) {
			cuda2_to_file( lyr.getGradWeights(), artifact_dir / "grad_linear_w.f32" );
			cuda2_to_file( lyr.getGradBias(), artifact_dir / "grad_linear_b.f32" );
		}
	} );
}

void dump_weights_after_step( TinyNN &nn, std::filesystem::path const &artifact_dir )
{
	nn.forEachLayer( [&]( auto &lyr ) {
		if constexpr ( requires { lyr.forEachLayer( []( auto & ) {} ); } ) {
			int conv_idx = 0;
			lyr.forEachLayer( [&]( auto &inner ) {
				if constexpr ( requires { inner.getWeights(); inner.kernelSize(); } ) {
					if ( conv_idx == 0 ) {
						cuda4_to_file( inner.getWeights(), artifact_dir / "conv1_w_after.f32" );
					} else {
						cuda4_to_file( inner.getWeights(), artifact_dir / "conv2_w_after.f32" );
					}
					++conv_idx;
				}
			} );
		} else if constexpr ( requires { lyr.getWeights(); lyr.outFeatures(); } ) {
			cuda2_to_file( lyr.getWeights(), artifact_dir / "linear_w_after.f32" );
		}
	} );
}

void apply_sgd_step( TinyNN &nn )
{
	nn.forEachLayer( [&]( auto &&layer ) {
		neural::detail::updateLayer_impl<Tensor2D_t>( layer, k_lr, k_weight_decay );
	} );
}

} // namespace

void run_tiny_conv2_linear_compare( std::filesystem::path const &artifact_dir )
{
	if ( !neural::cuda_runtime_ready() ) {
		std::cout << "tiny_conv2_linear_compare: no CUDA device.\n";
		return;
	}
	std::error_code ec;
	std::filesystem::create_directories( artifact_dir, ec );
	if ( ec ) {
		std::cout << "tiny_conv2_linear_compare: cannot create " << artifact_dir << "\n";
		return;
	}

	TinyBox box(
	    static_cast<std::size_t>( C ), static_cast<std::size_t>( H ), static_cast<std::size_t>( W ),
	    neural::ConvolutionalLayer<TensorN_t>( conv1_out, 3 ), neural::ReLULayer<TensorN_t>(),
	    neural::ConvolutionalLayer<TensorN_t>( conv2_out, 3 ), neural::ReLULayer<TensorN_t>() );

	TinyNN nn( box, neural::LinearLayer<Tensor2D_t>( linear_out ) );
	nn.ensureBuffersForShape( 1, in_cols );

	// Deterministic input (NCHW row-major as flattened CHW: c slow over h,w)
	std::vector<float> host_in( in_cols );
	for ( int c = 0; c < C; ++c ) {
		for ( int y = 0; y < H; ++y ) {
			for ( int x = 0; x < W; ++x ) {
				std::size_t const ix = static_cast<std::size_t>( c ) * static_cast<std::size_t>( H * W )
				                       + static_cast<std::size_t>( y * W + x );
				host_in[ix] = 0.01f * static_cast<float>( 1 + c * 11 + y * 3 + x );
			}
		}
	}
	write_f32_file( artifact_dir / "input.f32", host_in );

	std::vector<float> conv1_w;
	std::vector<float> conv2_w;
	std::vector<float> lin_w;
	std::mt19937_64 gen( k_global_seed );
	append_he( conv1_w, conv1_w_elems, 3u * 3u * 3u, gen );
	append_he( conv2_w, conv2_w_elems, conv1_out * 3u * 3u, gen );
	append_he( lin_w, linear_w_elems, flat_elems, gen );

	write_f32_file( artifact_dir / "conv1_w.f32", conv1_w );
	write_f32_file( artifact_dir / "conv2_w.f32", conv2_w );
	write_f32_file( artifact_dir / "linear_w.f32", lin_w );

	Tensor2D_t d_in( 1, in_cols );
	d_in.assign( host_in.data(), host_in.size() );
	(void)nn.forward( d_in );
	apply_host_weights( nn, conv1_w, conv2_w, lin_w );

	Tensor2D_t logits_dev = nn.forward( d_in );
	neural::cuda_layer_sync();
	neural::Tensor<float> logits( logits_dev.rows(), logits_dev.cols() );
	logits_dev.copyToHost( logits.data() );

	std::vector<float> logv( linear_out );
	for ( std::size_t j = 0; j < linear_out; ++j ) {
		logv[j] = logits( 0, j );
	}
	write_f32_file( artifact_dir / "logits_cpp.f32", logv );

	neural::Tensor<float> host_target( 1, linear_out, 0.f );
	host_target.assign( 0, target_class, 1.f );
	Tensor2D_t d_target( 1, linear_out );
	d_target.assign( host_target.data(), host_target.size() );

	float const loss = nn.trainStep( d_in, d_target );
	neural::cuda_layer_sync();
	{
		std::vector<float> one( 1u, loss );
		write_f32_file( artifact_dir / "train_loss_cpp.f32", one );
	}

	dump_gradients( nn, artifact_dir );
	apply_sgd_step( nn );
	neural::cuda_layer_sync();
	dump_weights_after_step( nn, artifact_dir );

	std::cout << std::setprecision( 9 ) << "tiny_conv2_linear_compare: wrote artifacts to " << artifact_dir
	          << "\n  CPP_LOGITS:";
	for ( float v : logv ) {
		std::cout << ' ' << v;
	}
	std::cout << "\n  CPP_TRAIN_LOSS: " << loss << '\n';
}

int main( int argc, char **argv )
{
	std::filesystem::path dir = "./tiny_compare_artifacts";
	if ( argc >= 2 ) {
		dir = argv[1];
	}
	run_tiny_conv2_linear_compare( dir );
	return 0;
}

#endif
