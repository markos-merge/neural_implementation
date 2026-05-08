#ifndef REFACTOR_CONVOLUTIONAL_LAYER_HPP
#define REFACTOR_CONVOLUTIONAL_LAYER_HPP

#include <array>
#include <cstddef>
#include <optional>
#include <random>
#include <stdexcept>

#include "device.hpp"
#include "layer_base.hpp" // refactor layer_base.hpp; avoid including layers/layer_base.hpp and this header in one TU (LayerBase conflict).
#include "cuda_tensor_n.hpp"
#include "neural_cuda_layer_sync.hpp"
#include "tensor_n.hpp"
#include <cuda_runtime.h>
#include <limits>
#include <type_traits>

namespace neural {
namespace refactor {

template <typename T, typename Device = Cpu>
class ConvolutionalLayer : public LayerBase<T, Device>
{
public:
	ConvolutionalLayer( std::size_t out_channels, std::size_t kernel_size,
	                    std::size_t cudnn_pad_h, std::size_t cudnn_pad_w );

	ConvolutionalLayer( std::size_t out_channels, std::size_t kernel_size,
	                    std::array<std::size_t, 3> input_chw,
	                    std::size_t cudnn_pad_h, std::size_t cudnn_pad_w );

	ConvolutionalLayer( T *weights, std::vector<std::size_t> shape, T *bias,
	                    std::size_t cudnn_pad_h, std::size_t cudnn_pad_w,
	                    std::optional<std::array<std::size_t, 3>> input_chw = std::nullopt );

	ConvolutionalLayer( std::size_t out_channels, std::size_t kernel_size )
	    : ConvolutionalLayer( out_channels, kernel_size, 0u, 0u )
	{
	}
	ConvolutionalLayer( std::size_t out_channels, std::size_t kernel_size,
	                    std::array<std::size_t, 3> input_chw )
	    : ConvolutionalLayer( out_channels, kernel_size, input_chw, 0u, 0u )
	{
	}

	~ConvolutionalLayer();

	ConvolutionalLayer( ConvolutionalLayer const &other );
	ConvolutionalLayer( ConvolutionalLayer &&other ) noexcept;
	ConvolutionalLayer &operator=( ConvolutionalLayer const &other );
	ConvolutionalLayer &operator=( ConvolutionalLayer &&other ) noexcept;

	std::size_t outChannels() const;
	std::size_t kernelSize() const;
	std::size_t cudnnPadH() const noexcept { return m_cuda_pad_h; }
	std::size_t cudnnPadW() const noexcept { return m_cuda_pad_w; }

	/// When set, the first conv sees a flat latch (N by C*H*W) and uses C,H,W here for geometry.
	std::optional<std::array<std::size_t, 3>> inputChw() const noexcept { return m_input_chw; }

	T *getWeights();
	T *getGradWeights();
	T *getBias();
	T *getGradBias();

	std::size_t numWeightParams() const noexcept { return m_weights.size(); }
	std::size_t numBiasParams() const noexcept { return m_bias.size(); }

	void forward();
	void backward();

private:
	void ensureWeights( std::size_t in_channels );

	using tensor_t =
	    std::conditional_t<std::is_same_v<Device, Cuda>, CudaTensor4<T>, TensorN<4, T>>;
	using bias_t           = tensor_t;
	using im2col_tensor_t = typename tensor_t::template tensor_alias<2, T>;

	std::size_t                               m_out_channels;
	std::size_t                               m_kernel_size;
	std::optional<std::array<std::size_t, 3>> m_input_chw;

	tensor_t m_weights;
	tensor_t m_grad_weights;
	bias_t   m_bias;
	bias_t   m_grad_bias;

	im2col_tensor_t m_im2col;
	tensor_t        m_gemm_cnhw;
	tensor_t        m_workspace_cnhw;

	void *      m_cudnn_workspace          = nullptr;
	std::size_t m_cudnn_workspace_capacity = 0;

	// CUDNN convolution padding (ignored on CPU — CPU path uses implicit K_h/2, K_w/2 in TensorN).
	std::size_t m_cuda_pad_h = 0;
	std::size_t m_cuda_pad_w = 0;
};

// =============================================================================
// Implementation
// =============================================================================

template <typename T, typename Device>
ConvolutionalLayer<T, Device>::ConvolutionalLayer( std::size_t out_channels,
                                                   std::size_t kernel_size,
                                                   std::size_t cudnn_pad_h,
                                                   std::size_t cudnn_pad_w )
    : m_out_channels( out_channels ), m_kernel_size( kernel_size ),
      m_cuda_pad_h( cudnn_pad_h ), m_cuda_pad_w( cudnn_pad_w )
{
}

template <typename T, typename Device>
ConvolutionalLayer<T, Device>::ConvolutionalLayer(
    std::size_t out_channels, std::size_t kernel_size,
    std::array<std::size_t, 3> input_chw, std::size_t cudnn_pad_h,
    std::size_t cudnn_pad_w )
    : m_out_channels( out_channels ), m_kernel_size( kernel_size ),
      m_input_chw( input_chw ), m_cuda_pad_h( cudnn_pad_h ),
      m_cuda_pad_w( cudnn_pad_w )
{
	this->ensureWeights( input_chw[0] );
}

template <typename T, typename Device>
ConvolutionalLayer<T, Device>::ConvolutionalLayer(
    T *weights, std::vector<std::size_t> shape, T *bias,
    std::size_t cudnn_pad_h, std::size_t cudnn_pad_w,
    std::optional<std::array<std::size_t, 3>> input_chw )
    : m_out_channels( shape[0] )
    , m_kernel_size( shape[2] )
    , m_input_chw( std::move( input_chw ) )
    , m_weights( weights, shape, true )
    , m_bias( bias, std::array<std::size_t, 4>{ 1, shape[0], 1, 1 }, true )
    , m_cuda_pad_h( cudnn_pad_h )
    , m_cuda_pad_w( cudnn_pad_w )
{
	if ( m_input_chw ) {
		if ( ( *m_input_chw )[0] != shape[1] ) {
			throw std::invalid_argument(
			    "ConvolutionalLayer(weights,...): input_chw C does not match weight in_channels" );
		}
	}
	m_grad_weights = tensor_t( m_weights.shape() );
	m_grad_bias    = bias_t( m_bias.shape() );
}

template <typename T, typename Device>
ConvolutionalLayer<T, Device>::~ConvolutionalLayer()
{
	if constexpr ( std::is_same_v<Device, Cuda> ) {
		if ( m_cudnn_workspace != nullptr ) {
			(void)cudaFree( m_cudnn_workspace );
			m_cudnn_workspace          = nullptr;
			m_cudnn_workspace_capacity = 0;
		}
	}
}

template <typename T, typename Device>
ConvolutionalLayer<T, Device>::ConvolutionalLayer( ConvolutionalLayer const &other )
	: LayerBase<T, Device>( other )
	, m_out_channels( other.m_out_channels )
	, m_kernel_size( other.m_kernel_size )
	, m_input_chw( other.m_input_chw )
	, m_weights( other.m_weights )
	, m_bias( other.m_bias )
	, m_grad_weights( other.m_grad_weights )
	, m_grad_bias( other.m_grad_bias )
	, m_im2col( other.m_im2col )
	, m_gemm_cnhw( other.m_gemm_cnhw )
	, m_workspace_cnhw( other.m_workspace_cnhw )
	, m_cudnn_workspace( nullptr )
	, m_cudnn_workspace_capacity( 0 )
	, m_cuda_pad_h( other.m_cuda_pad_h )
	, m_cuda_pad_w( other.m_cuda_pad_w )
{
}

template <typename T, typename Device>
ConvolutionalLayer<T, Device>::ConvolutionalLayer( ConvolutionalLayer &&other ) noexcept
	: LayerBase<T, Device>( std::move( other ) )
	, m_out_channels( other.m_out_channels )
	, m_kernel_size( other.m_kernel_size )
	, m_input_chw( std::move( other.m_input_chw ) )
	, m_weights( std::move( other.m_weights ) )
	, m_bias( std::move( other.m_bias ) )
	, m_grad_weights( std::move( other.m_grad_weights ) )
	, m_grad_bias( std::move( other.m_grad_bias ) )
	, m_im2col( std::move( other.m_im2col ) )
	, m_gemm_cnhw( std::move( other.m_gemm_cnhw ) )
	, m_workspace_cnhw( std::move( other.m_workspace_cnhw ) )
	, m_cudnn_workspace( other.m_cudnn_workspace )
	, m_cudnn_workspace_capacity( other.m_cudnn_workspace_capacity )
	, m_cuda_pad_h( other.m_cuda_pad_h )
	, m_cuda_pad_w( other.m_cuda_pad_w )
{
	other.m_cudnn_workspace          = nullptr;
	other.m_cudnn_workspace_capacity = 0;
}

template <typename T, typename Device>
ConvolutionalLayer<T, Device> &ConvolutionalLayer<T, Device>::operator=( ConvolutionalLayer const &other )
{
	if ( this != &other ) {
		if constexpr ( std::is_same_v<Device, Cuda> ) {
			if ( m_cudnn_workspace != nullptr ) {
				(void)cudaFree( m_cudnn_workspace );
				m_cudnn_workspace          = nullptr;
				m_cudnn_workspace_capacity = 0;
			}
		}
		LayerBase<T, Device>::operator=( other );
		m_out_channels   = other.m_out_channels;
		m_kernel_size    = other.m_kernel_size;
		m_input_chw      = other.m_input_chw;
		m_weights        = other.m_weights;
		m_bias           = other.m_bias;
		m_grad_weights   = other.m_grad_weights;
		m_grad_bias      = other.m_grad_bias;
		m_im2col         = other.m_im2col;
		m_gemm_cnhw      = other.m_gemm_cnhw;
		m_workspace_cnhw = other.m_workspace_cnhw;
		m_cuda_pad_h = other.m_cuda_pad_h;
		m_cuda_pad_w = other.m_cuda_pad_w;
	}
	return *this;
}

template <typename T, typename Device>
ConvolutionalLayer<T, Device> &ConvolutionalLayer<T, Device>::operator=( ConvolutionalLayer &&other ) noexcept
{
	if ( this != &other ) {
		if constexpr ( std::is_same_v<Device, Cuda> ) {
			if ( m_cudnn_workspace != nullptr ) {
				(void)cudaFree( m_cudnn_workspace );
			}
		}
		LayerBase<T, Device>::operator=( std::move( other ) );
		m_out_channels   = other.m_out_channels;
		m_kernel_size    = other.m_kernel_size;
		m_input_chw      = std::move( other.m_input_chw );
		m_weights        = std::move( other.m_weights );
		m_bias           = std::move( other.m_bias );
		m_grad_weights   = std::move( other.m_grad_weights );
		m_grad_bias      = std::move( other.m_grad_bias );
		m_im2col         = std::move( other.m_im2col );
		m_gemm_cnhw      = std::move( other.m_gemm_cnhw );
		m_workspace_cnhw = std::move( other.m_workspace_cnhw );
		m_cuda_pad_h     = other.m_cuda_pad_h;
		m_cuda_pad_w     = other.m_cuda_pad_w;
		m_cudnn_workspace          = other.m_cudnn_workspace;
		m_cudnn_workspace_capacity = other.m_cudnn_workspace_capacity;
		other.m_cudnn_workspace          = nullptr;
		other.m_cudnn_workspace_capacity = 0;
	}
	return *this;
}

template <typename T, typename Device>
std::size_t ConvolutionalLayer<T, Device>::outChannels() const
{
	return m_out_channels;
}

template <typename T, typename Device>
std::size_t ConvolutionalLayer<T, Device>::kernelSize() const
{
	return m_kernel_size;
}

template <typename T, typename Device>
void ConvolutionalLayer<T, Device>::ensureWeights( std::size_t in_channels )
{
	if ( m_weights.size() > 0 ) {
		if ( m_weights.shape()[1] != in_channels ) {
			throw std::logic_error(
			    "RefactorConvolutionalLayer: input channel count does not match allocated weights" );
		}
		return;
	}
	std::array<std::size_t, 4> const wshape{ m_out_channels, in_channels, m_kernel_size,
	                                         m_kernel_size };
	m_weights      = tensor_t( wshape );
	m_bias         = bias_t( { 1, m_out_channels, 1, 1 } );
	m_grad_weights = tensor_t( wshape );
	m_grad_bias    = bias_t( { 1, m_out_channels, 1, 1 } );

	auto const &ws    = m_weights.shape();
	std::size_t const fan_in = ws[1] * ws[2] * ws[3];
	std::mt19937_64     wgen( std::random_device{}() );
	std::uint64_t const s = wgen();
	m_weights.randomizeHe( fan_in, s );
}

template <typename T, typename Device>
T *ConvolutionalLayer<T, Device>::getWeights()
{
	return m_weights.data();
}

template <typename T, typename Device>
T *ConvolutionalLayer<T, Device>::getGradWeights()
{
	return m_grad_weights.data();
}

template <typename T, typename Device>
T *ConvolutionalLayer<T, Device>::getBias()
{
	return m_bias.data();
}

template <typename T, typename Device>
T *ConvolutionalLayer<T, Device>::getGradBias()
{
	return m_grad_bias.data();
}

template <typename T, typename Device>
void ConvolutionalLayer<T, Device>::forward()
{
	std::vector<std::size_t> const &in_vec = this->getInput()->shape();
	if ( in_vec.size() != 4 && !m_input_chw ) {
		throw std::invalid_argument( "RefactorConvolutionalLayer::forward: input must be rank 4 (NCHW)" );
	}

	std::array<std::size_t, 4> const in_shape = [&]() {
		std::array<std::size_t, 4> ret{ 0, 0, 0, 0 };
		if ( m_input_chw ) {
			ret[0] = in_vec[0];
			ret[1] = m_input_chw->at( 0 );
			ret[2] = m_input_chw->at( 1 );
			ret[3] = m_input_chw->at( 2 );
		} else {
			std::copy( in_vec.begin(), in_vec.end(), ret.begin() );
		}
		return ret;
	}();
	std::size_t const in_channels = in_shape[1];
	this->ensureWeights( in_channels );

	std::size_t H_out;
	std::size_t W_out;

	if constexpr ( std::is_same_v<Device, Cpu> ) {
		std::size_t const pad_i = m_kernel_size / 2;
		std::size_t const pad_j = m_kernel_size / 2;
		H_out = ( in_shape[2] > 2 * pad_i ) ? in_shape[2] - 2 * pad_i : in_shape[2];
		W_out = ( in_shape[3] > 2 * pad_j ) ? in_shape[3] - 2 * pad_j : in_shape[3];
	} else {
		std::size_t const ph = m_cuda_pad_h;
		std::size_t const pw = m_cuda_pad_w;
		auto const         ws_sh = m_weights.shape();
		std::size_t const  Kh = ws_sh[2];
		std::size_t const  Kw = ws_sh[3];
		std::size_t const  sum_h = in_shape[2] + 2 * ph;
		std::size_t const  sum_w = in_shape[3] + 2 * pw;
		if ( sum_h < Kh || sum_w < Kw ) {
			throw std::invalid_argument(
			    "RefactorConvolutionalLayer::forward: input spatial size incompatible with CUDNN padding and kernel" );
		}
		H_out = 1 + sum_h - Kh;
		W_out = 1 + sum_w - Kw;
		if ( H_out == 0 || W_out == 0 ) {
			throw std::invalid_argument( "RefactorConvolutionalLayer::forward: CUDNN output extent is zero" );
		}
	}

	std::vector<std::size_t> const out_vec{ in_shape[0], m_out_channels, H_out, W_out };

	if ( this->getOutput()->shape() != out_vec ) {
		this->getOutput()->resize( out_vec );
	}

	tensor_t input_view( this->getInput()->fwdData(), in_shape, false );
	tensor_t output_view( this->getOutput()->fwdData(), out_vec, false );

	if constexpr ( std::is_same_v<Device, Cpu> ) {
		std::array<std::size_t, 2> const im2_shape{ in_channels * m_kernel_size * m_kernel_size,
		                                             in_shape[0] * H_out * W_out };
		if ( m_im2col.shape() != im2_shape ) {
			m_im2col = im2col_tensor_t( im2_shape );
		}

		input_view.im2ColConvolution( m_weights, m_im2col, output_view, m_gemm_cnhw );
		output_view.addColorChannelInPlace( m_bias );
	} else {
		if ( ( m_kernel_size % 2 ) == 0 ) {
			throw std::invalid_argument(
			    "RefactorConvolutionalLayer: CUDA cuDNN path requires an odd kernel size" );
		}

		if ( m_cuda_pad_h > static_cast<std::size_t>(
			 std::numeric_limits<int>::max() ) ||
		     m_cuda_pad_w > static_cast<std::size_t>(
			 std::numeric_limits<int>::max() ) ) {
			throw std::invalid_argument(
			    "RefactorConvolutionalLayer: CUDNN padding exceeds int representable range" );
		}

		input_view.im2ColConvolution(
		    m_weights, m_im2col, output_view, m_gemm_cnhw, &m_cudnn_workspace,
		    &m_cudnn_workspace_capacity, static_cast<int>( m_cuda_pad_h ),
		    static_cast<int>( m_cuda_pad_w ) );
		output_view.addColorChannelInPlace( m_bias );
		cuda_layer_sync();
	}
}

template <typename T, typename Device>
void ConvolutionalLayer<T, Device>::backward()
{
	if ( m_weights.size() == 0 ) {
		throw std::logic_error( "RefactorConvolutionalLayer::backward: forward has not been run" );
	}
	std::vector<std::size_t> const &in_vec = this->getInput()->shape();
	std::array<std::size_t, 4> const in_shape = [&]() {
		std::array<std::size_t, 4> ret{ 0, 0, 0, 0 };
		if ( m_input_chw ) {
			ret[0] = in_vec[0];
			ret[1] = m_input_chw->at( 0 );
			ret[2] = m_input_chw->at( 1 );
			ret[3] = m_input_chw->at( 2 );
		} else {
			std::copy( in_vec.begin(), in_vec.end(), ret.begin() );
		}
		return ret;
	}();
	if ( !std::equal(this->getGradOutput()->shape().begin(), this->getGradOutput()->shape().end(), in_shape.begin(), in_shape.end() ) ) {
		std::vector<std::size_t> temp;
		std::copy( in_shape.begin(), in_shape.end(), std::back_inserter( temp ) );
		this->getGradOutput()->resize( temp );
	}

	if constexpr ( std::is_same_v<Device, Cpu> ) {
		tensor_t g_dy_view( this->getGradInput()->fwdData(),
		                    this->getGradInput()->shape(), false );
		tensor_t gx_view( this->getGradOutput()->fwdData(), in_shape, false );

		convolutional_backward_im2col(
		    g_dy_view, m_weights, m_im2col, in_shape, m_grad_weights, m_grad_bias, gx_view,
		    m_workspace_cnhw );
	} else {
		tensor_t g_dy_view( this->getGradInput()->fwdData(),
		                    this->getGradInput()->shape(), false );
		tensor_t input_view( this->getInput()->fwdData(), in_shape, false );
		tensor_t gx_view( this->getGradOutput()->fwdData(), in_shape, false );

		convolutional_backward_cudnn( g_dy_view, m_weights, input_view, m_grad_weights, m_grad_bias,
		                              gx_view, &m_cudnn_workspace, &m_cudnn_workspace_capacity,
		                              static_cast<int>( m_cuda_pad_h ), static_cast<int>( m_cuda_pad_w ) );
		cuda_layer_sync();
	}
}

} // namespace refactor
} // namespace neural

#endif // REFACTOR_CONVOLUTIONAL_LAYER_HPP
