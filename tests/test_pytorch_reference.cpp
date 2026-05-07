// Compare CPU/CUDA/cuDNN against reference tensors from PyTorch
// (scripts/generate_pytorch_reference.py → tests/data/pytorch_ref/{mlp,conv2d}/).

#include "layer_wiring_helpers.hpp"
#include "linear_layer.hpp"
#include "mse_loss.hpp"
#include "neural_cuda_runtime.hpp"
#include "relu_layer.hpp"
#include "sequential_nn_static.hpp"
#include "tensor.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#if NEURAL_CUDA_ENABLED
#include "cuda_tensor.hpp"
#endif
#if NEURAL_CUDNN_ENABLED
#include "convolutional_layer.hpp"
#include "cuda_tensor_n.hpp"
#include <cuda_runtime.h>
#endif

using neural::LinearLayer;
using neural::MSELoss;
using neural::ReLULayer;
using neural::SequentialNN_static;
using neural::Tensor;
using Catch::Matchers::WithinAbs;

namespace {

constexpr float k_eps     = 1e-4f;
constexpr float k_eps_bwd = 2e-3f; // a bit looser (CUDA + accumulation)

// tests/data/pytorch_ref/... next to this file
std::filesystem::path pytorch_ref_mlp_dir()
{
	return std::filesystem::path( __FILE__ ).parent_path() / "data" / "pytorch_ref" / "mlp";
}

#if NEURAL_CUDNN_ENABLED
std::filesystem::path pytorch_ref_conv2d_dir()
{
	return std::filesystem::path( __FILE__ ).parent_path() / "data" / "pytorch_ref" / "conv2d";
}
#endif

std::vector<float> read_float_file( std::filesystem::path const &path )
{
	std::ifstream f( path, std::ios::binary );
	REQUIRE( f.good() );
	f.seekg( 0, std::ios::end );
	auto const n = static_cast<std::size_t>( f.tellg() );
	REQUIRE( n % sizeof( float ) == 0 );
	f.seekg( 0, std::ios::beg );
	std::vector<float> out( n / sizeof( float ) );
	f.read( reinterpret_cast<char *>( out.data() ),
	        static_cast<std::streamsize>( n ) );
	REQUIRE( static_cast<std::size_t>( f.gcount() ) == n );
	return out;
}

void require_close_tensor( Tensor<float> const &a, std::vector<float> const &ref, float eps )
{
	REQUIRE( a.size() == ref.size() );
	for ( std::size_t i = 0; i < ref.size(); ++i ) {
		REQUIRE_THAT( a.data()[i], WithinAbs( ref[i], eps ) );
	}
}

#if NEURAL_CUDNN_ENABLED
std::vector<float> read_all_floats( std::filesystem::path const &path )
{
	return read_float_file( path );
}

struct Meta
{
	std::size_t n = 0, ci = 0, h = 0, w = 0, co = 0, k = 0, oh = 0, ow = 0;
};

Meta read_meta( std::filesystem::path const &p )
{
	std::ifstream f( p );
	REQUIRE( f.good() );
	std::string line;
	REQUIRE( std::getline( f, line ) );
	std::istringstream iss( line );
	Meta m;
	REQUIRE( iss >> m.n >> m.ci >> m.h >> m.w >> m.co >> m.k >> m.oh >> m.ow );
	return m;
}

#endif

} // namespace

#if NEURAL_CUDNN_ENABLED
namespace {

using neural::CudaTensor4;

std::vector<float> conv_to_host( CudaTensor4<float> const &t )
{
	std::vector<float> v( t.size() );
	if ( !v.empty() &&
	     cudaMemcpy( v.data(), t.data(), v.size() * sizeof( float ), cudaMemcpyDeviceToHost ) !=
	         cudaSuccess ) {
		throw std::runtime_error( "pytorch ref conv: device to host copy failed" );
	}
	return v;
}

} // namespace
#endif

// ——— MLP ———

TEST_CASE( "PyTorch ref tiny MLP forward (CPU)", "[pytorch_reference][cpu]" )
{
	std::filesystem::path const d = pytorch_ref_mlp_dir();
	std::vector<float> const w0  = read_float_file( d / "w0.bin" );
	std::vector<float> const b0  = read_float_file( d / "b0.bin" );
	std::vector<float> const w1  = read_float_file( d / "w1.bin" );
	std::vector<float> const b1  = read_float_file( d / "b1.bin" );
	std::vector<float> const xb  = read_float_file( d / "x.bin" );
	std::vector<float> const yref = read_float_file( d / "y_ref.bin" );

	REQUIRE( w0.size() == 4u * 3u );
	REQUIRE( b0.size() == 3u );
	REQUIRE( w1.size() == 3u * 2u );
	REQUIRE( b1.size() == 2u );
	REQUIRE( xb.size() == 2u * 4u );
	REQUIRE( yref.size() == 2u * 2u );

	Tensor<float> W0( 4, 3, w0.begin(), w0.end() );
	Tensor<float> B0( 3, 1, b0.begin(), b0.end() );
	Tensor<float> W1( 3, 2, w1.begin(), w1.end() );
	Tensor<float> B1( 2, 1, b1.begin(), b1.end() );

	LinearLayer<Tensor<float>> fc0( W0, B0 );
	ReLULayer<Tensor<float>>      relu;
	LinearLayer<Tensor<float>>     fc1( W1, B1 );
	SequentialNN_static<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>,
	             ReLULayer<Tensor<float>>, LinearLayer<Tensor<float>>>
	    nn( fc0, relu, fc1 );

	Tensor<float> input( 2, 4, xb.begin(), xb.end() );
	Tensor<float> out = nn.forward( input );
	REQUIRE( out.rows() == 2u );
	REQUIRE( out.cols() == 2u );
	for ( std::size_t r = 0; r < 2; ++r ) {
		for ( std::size_t c = 0; c < 2; ++c ) {
			float const e = yref[r * 2u + c];
			REQUIRE_THAT( out( r, c ), WithinAbs( e, k_eps ) );
		}
	}
}

#if NEURAL_CUDA_ENABLED
TEST_CASE( "PyTorch ref tiny MLP forward (CUDA)", "[pytorch_reference][cuda]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::filesystem::path const d = pytorch_ref_mlp_dir();
	std::vector<float> const w0  = read_float_file( d / "w0.bin" );
	std::vector<float> const b0  = read_float_file( d / "b0.bin" );
	std::vector<float> const w1  = read_float_file( d / "w1.bin" );
	std::vector<float> const b1  = read_float_file( d / "b1.bin" );
	std::vector<float> const xb  = read_float_file( d / "x.bin" );
	std::vector<float> const yref = read_float_file( d / "y_ref.bin" );

	using T = float;
	neural::CudaTensor<T> W0( 4, 3, w0.begin(), w0.end() );
	neural::CudaTensor<T> B0( 3, 1, b0.begin(), b0.end() );
	neural::CudaTensor<T> W1( 3, 2, w1.begin(), w1.end() );
	neural::CudaTensor<T> B1( 2, 1, b1.begin(), b1.end() );

	LinearLayer<neural::CudaTensor<T>> fc0( W0, B0 );
	ReLULayer<neural::CudaTensor<T>>    relu;
	LinearLayer<neural::CudaTensor<T>>  fc1( W1, B1 );
	SequentialNN_static<neural::CudaTensor<T>, MSELoss<neural::CudaTensor<T>>,
	             LinearLayer<neural::CudaTensor<T>>, ReLULayer<neural::CudaTensor<T>>,
	             LinearLayer<neural::CudaTensor<T>>>
	    nn( fc0, relu, fc1 );

	neural::CudaTensor<T> input( 2, 4, xb.begin(), xb.end() );
	neural::CudaTensor<T> out = nn.forward( input );
	REQUIRE( out.rows() == 2u );
	REQUIRE( out.cols() == 2u );
	std::vector<T> ygot( 2u * 2u );
	out.copyToHost( ygot.data() );
	for ( std::size_t i = 0; i < 4; ++i ) {
		REQUIRE_THAT( ygot[i], WithinAbs( yref[i], k_eps ) );
	}
}
#endif

TEST_CASE( "PyTorch ref tiny MLP backward MSE (CPU)", "[pytorch_reference][cpu][backward]" )
{
	std::filesystem::path const d = pytorch_ref_mlp_dir();
	std::vector<float> const w0  = read_float_file( d / "w0.bin" );
	std::vector<float> const b0  = read_float_file( d / "b0.bin" );
	std::vector<float> const w1  = read_float_file( d / "w1.bin" );
	std::vector<float> const b1  = read_float_file( d / "b1.bin" );
	std::vector<float> const xb  = read_float_file( d / "x.bin" );
	std::vector<float> const tgt = read_float_file( d / "target.bin" );
	std::vector<float> const gW0 = read_float_file( d / "grad_w0.bin" );
	std::vector<float> const gB0 = read_float_file( d / "grad_b0.bin" );
	std::vector<float> const gW1 = read_float_file( d / "grad_w1.bin" );
	std::vector<float> const gB1 = read_float_file( d / "grad_b1.bin" );
	std::vector<float> const gX  = read_float_file( d / "grad_x.bin" );

	Tensor<float> W0( 4, 3, w0.begin(), w0.end() );
	Tensor<float> B0( 3, 1, b0.begin(), b0.end() );
	Tensor<float> W1( 3, 2, w1.begin(), w1.end() );
	Tensor<float> B1( 2, 1, b1.begin(), b1.end() );

	SequentialNN_static<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>,
	             ReLULayer<Tensor<float>>, LinearLayer<Tensor<float>>>
	    nn( LinearLayer<Tensor<float>>( W0, B0 ),
	        ReLULayer<Tensor<float>>(),
	        LinearLayer<Tensor<float>>( W1, B1 ) );

	Tensor<float> input( 2, 4, xb.begin(), xb.end() );
	Tensor<float> target( 2, 2, tgt.begin(), tgt.end() );
	(void)nn.trainStep( input, target );

	int linear_index = 0;
	nn.forEachLayer( [&]( auto &layer ) {
		using L = std::decay_t<decltype( layer )>;
		if constexpr ( std::is_same_v<L, LinearLayer<Tensor<float>>> ) {
			if ( linear_index == 0 ) {
				require_close_tensor( layer.getGradWeights(), gW0, k_eps_bwd );
				require_close_tensor( layer.getGradBias(), gB0, k_eps_bwd );
			} else {
				require_close_tensor( layer.getGradWeights(), gW1, k_eps_bwd );
				require_close_tensor( layer.getGradBias(), gB1, k_eps_bwd );
			}
			++linear_index;
		}
	} );
	require_close_tensor( *nn.inputLatch().grads(), gX, k_eps_bwd );
}

#if NEURAL_CUDA_ENABLED
TEST_CASE( "PyTorch ref tiny MLP backward MSE (CUDA)", "[pytorch_reference][cuda][backward]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::filesystem::path const d = pytorch_ref_mlp_dir();
	std::vector<float> const w0  = read_float_file( d / "w0.bin" );
	std::vector<float> const b0  = read_float_file( d / "b0.bin" );
	std::vector<float> const w1  = read_float_file( d / "w1.bin" );
	std::vector<float> const b1  = read_float_file( d / "b1.bin" );
	std::vector<float> const xb  = read_float_file( d / "x.bin" );
	std::vector<float> const tgt = read_float_file( d / "target.bin" );
	std::vector<float> const gW0 = read_float_file( d / "grad_w0.bin" );
	std::vector<float> const gB0 = read_float_file( d / "grad_b0.bin" );
	std::vector<float> const gW1 = read_float_file( d / "grad_w1.bin" );
	std::vector<float> const gB1 = read_float_file( d / "grad_b1.bin" );
	std::vector<float> const gX  = read_float_file( d / "grad_x.bin" );

	using T = float;
	neural::CudaTensor<T> W0( 4, 3, w0.begin(), w0.end() );
	neural::CudaTensor<T> B0( 3, 1, b0.begin(), b0.end() );
	neural::CudaTensor<T> W1( 3, 2, w1.begin(), w1.end() );
	neural::CudaTensor<T> B1( 2, 1, b1.begin(), b1.end() );

	SequentialNN_static<neural::CudaTensor<T>, MSELoss<neural::CudaTensor<T>>,
	             LinearLayer<neural::CudaTensor<T>>, ReLULayer<neural::CudaTensor<T>>,
	             LinearLayer<neural::CudaTensor<T>>>
	    nn( LinearLayer<neural::CudaTensor<T>>( W0, B0 ),
	        ReLULayer<neural::CudaTensor<T>>(),
	        LinearLayer<neural::CudaTensor<T>>( W1, B1 ) );

	neural::CudaTensor<T> input( 2, 4, xb.begin(), xb.end() );
	neural::CudaTensor<T> target( 2, 2, tgt.begin(), tgt.end() );
	(void)nn.trainStep( input, target );

	int linear_index = 0;
	nn.forEachLayer( [&]( auto &layer ) {
		using L = std::decay_t<decltype( layer )>;
		if constexpr ( std::is_same_v<L, LinearLayer<neural::CudaTensor<T>>> ) {
			std::vector<T> h;
			if ( linear_index == 0 ) {
				h.resize( gW0.size() );
				layer.getGradWeights().copyToHost( h.data() );
				for ( std::size_t i = 0; i < gW0.size(); ++i ) {
					REQUIRE_THAT( h[i], WithinAbs( gW0[i], k_eps_bwd ) );
				}
				h.assign( gB0.size(), T{} );
				layer.getGradBias().copyToHost( h.data() );
				for ( std::size_t i = 0; i < gB0.size(); ++i ) {
					REQUIRE_THAT( h[i], WithinAbs( gB0[i], k_eps_bwd ) );
				}
			} else {
				h.resize( gW1.size() );
				layer.getGradWeights().copyToHost( h.data() );
				for ( std::size_t i = 0; i < gW1.size(); ++i ) {
					REQUIRE_THAT( h[i], WithinAbs( gW1[i], k_eps_bwd ) );
				}
				h.assign( gB1.size(), T{} );
				layer.getGradBias().copyToHost( h.data() );
				for ( std::size_t i = 0; i < gB1.size(); ++i ) {
					REQUIRE_THAT( h[i], WithinAbs( gB1[i], k_eps_bwd ) );
				}
			}
			++linear_index;
		}
	} );
	std::vector<T> hx( gX.size() );
	nn.inputLatch().grads()->copyToHost( hx.data() );
	for ( std::size_t i = 0; i < gX.size(); ++i ) {
		REQUIRE_THAT( hx[i], WithinAbs( gX[i], k_eps_bwd ) );
	}
}
#endif

// ——— Single Conv2d (cuDNN) ———
// im2col CPU path can disagree with "same" padding; this checks CUDA/cuDNN only.

#if NEURAL_CUDNN_ENABLED

using neural::ConvolutionalLayer;
using neural::CudaTensor4;
using neural::test::wire_layer;

namespace {
constexpr float k_conv_eps = 2e-3f;
} // namespace

TEST_CASE( "PyTorch ref single Conv2d forward (CUDA/cuDNN)", "[pytorch_reference][conv][cuda]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::filesystem::path const d = pytorch_ref_conv2d_dir();
	Meta const meta    = read_meta( d / "meta.txt" );
	std::vector<float> const x_h   = read_all_floats( d / "x.bin" );
	std::vector<float> const w_h   = read_all_floats( d / "w.bin" );
	std::vector<float> const b_h   = read_all_floats( d / "b.bin" );
	std::vector<float> const y_ref = read_all_floats( d / "y_ref.bin" );

	std::array<std::size_t, 4> const in_shape{ meta.n, meta.ci, meta.h, meta.w };
	std::array<std::size_t, 4> const out_shape{ meta.n, meta.co, meta.oh, meta.ow };
	REQUIRE( x_h.size() == meta.n * meta.ci * meta.h * meta.w );
	REQUIRE( w_h.size() == meta.co * meta.ci * meta.k * meta.k );
	REQUIRE( b_h.size() == meta.co );
	REQUIRE( y_ref.size() == out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] );

	ConvolutionalLayer<CudaTensor4<float>> layer( meta.co, meta.k );
	CudaTensor4<float> in_buf( in_shape );
	in_buf.assign( x_h );
	CudaTensor4<float> out_buf( out_shape, 0.f );
	CudaTensor4<float> g_out( out_shape, 0.f );
	CudaTensor4<float> g_in( in_shape, 0.f );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );

	layer.forward();
	layer.getWeights().assign( w_h );
	layer.getBias().assign( b_h );
	layer.forward();

	std::vector<float> const y_gpu = conv_to_host( out_buf );
	REQUIRE( y_gpu.size() == y_ref.size() );
	for ( std::size_t i = 0; i < y_ref.size(); ++i ) {
		REQUIRE_THAT( y_gpu[i], WithinAbs( y_ref[i], k_conv_eps ) );
	}
}

TEST_CASE( "PyTorch ref single Conv2d backward (CUDA/cuDNN)", "[pytorch_reference][conv][cuda][backward]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::filesystem::path const d = pytorch_ref_conv2d_dir();
	Meta const meta     = read_meta( d / "meta.txt" );
	std::vector<float> const x_h   = read_all_floats( d / "x.bin" );
	std::vector<float> const w_h   = read_all_floats( d / "w.bin" );
	std::vector<float> const b_h   = read_all_floats( d / "b.bin" );
	std::vector<float> const g_y_h = read_all_floats( d / "g_y.bin" );
	std::vector<float> const gWref = read_all_floats( d / "grad_w.bin" );
	std::vector<float> const gBref = read_all_floats( d / "grad_b.bin" );
	std::vector<float> const gXref = read_all_floats( d / "grad_x.bin" );

	std::array<std::size_t, 4> const in_shape{ meta.n, meta.ci, meta.h, meta.w };
	std::array<std::size_t, 4> const out_shape{ meta.n, meta.co, meta.oh, meta.ow };
	REQUIRE( x_h.size() == meta.n * meta.ci * meta.h * meta.w );
	REQUIRE( w_h.size() == meta.co * meta.ci * meta.k * meta.k );
	REQUIRE( b_h.size() == meta.co );
	REQUIRE( g_y_h.size() == out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] );
	REQUIRE( gWref.size() == w_h.size() );
	REQUIRE( gBref.size() == b_h.size() );
	REQUIRE( gXref.size() == x_h.size() );

	ConvolutionalLayer<CudaTensor4<float>> layer( meta.co, meta.k );
	CudaTensor4<float> in_buf( in_shape );
	in_buf.assign( x_h );
	CudaTensor4<float> out_buf( out_shape, 0.f );
	CudaTensor4<float> g_out( out_shape, 0.f );
	CudaTensor4<float> g_in( in_shape, 0.f );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );

	layer.forward();
	layer.getWeights().assign( w_h );
	layer.getBias().assign( b_h );
	layer.forward();

	g_out.assign( g_y_h );
	layer.backward();

	std::vector<float> const gw = conv_to_host( layer.getGradWeights() );
	std::vector<float> const gx = conv_to_host( g_in );
	std::vector<float> const gb = conv_to_host( layer.getGradBias() );

	REQUIRE( gw.size() == gWref.size() );
	for ( std::size_t i = 0; i < gw.size(); ++i ) {
		REQUIRE_THAT( gw[i], WithinAbs( gWref[i], k_conv_eps ) );
	}
	REQUIRE( gb.size() == gBref.size() );
	for ( std::size_t c = 0; c < gBref.size(); ++c ) {
		REQUIRE_THAT( gb[c], WithinAbs( gBref[c], k_conv_eps ) );
	}
	REQUIRE( gx.size() == gXref.size() );
	for ( std::size_t i = 0; i < gx.size(); ++i ) {
		REQUIRE_THAT( gx[i], WithinAbs( gXref[i], k_conv_eps ) );
	}
}

#endif // NEURAL_CUDNN_ENABLED
