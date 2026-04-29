#ifndef CONVOLUTIONAL_LAYER_HPP
#define CONVOLUTIONAL_LAYER_HPP

#include <cstddef>
#include <array>
#include <random>
#include <stdexcept>
#include "layer_base.hpp"
#include "tensor_n.hpp"
#if NEURAL_CUDA_ENABLED
#include "cuda_tensor_n.hpp"
#include "neural_cuda_layer_sync.hpp"
#endif
#if NEURAL_CUDNN_ENABLED
#include <cuda_runtime.h>
#endif

namespace neural {

template <typename TensorN_t>
class ConvolutionalLayer : public LayerBase< TensorN_t >
{
	public:
		using tensor_t = TensorN_t;
		using bias_t = TensorN_t;

		/// Only output channels and kernel; input channel count is fixed on first forward from NCHW input.
		ConvolutionalLayer( std::size_t out_channels, std::size_t kernel_size );

#if NEURAL_CUDNN_ENABLED
		~ConvolutionalLayer();
		ConvolutionalLayer( ConvolutionalLayer const &other );
		ConvolutionalLayer( ConvolutionalLayer &&other ) noexcept;
		ConvolutionalLayer &operator=( ConvolutionalLayer const &other );
		ConvolutionalLayer &operator=( ConvolutionalLayer &&other ) noexcept;
#endif

		std::size_t outChannels() const { return m_out_channels; }
		std::size_t kernelSize() const { return m_kernel_size; }

		void initialize();

		TensorN_t *forward();
		TensorN_t *backward();

		TensorN_t       &getWeights()     { return m_weights; }
		TensorN_t const &getWeights()     const { return m_weights; }
		bias_t       &getBias()        { return m_bias; }
		bias_t const &getBias()        const { return m_bias; }
		TensorN_t       &getGradWeights() { return m_grad_weights; }
		TensorN_t const &getGradWeights() const { return m_grad_weights; }
		bias_t       &getGradBias()    { return m_grad_bias; }
		bias_t const &getGradBias()    const { return m_grad_bias; }

	private:
		void ensure_weights( std::size_t in_channels );

		using aux_tensor_type = typename TensorN_t::template tensor_alias<
		    2, typename TensorN_t::value_type>;
		std::size_t m_out_channels;
		std::size_t m_kernel_size;

		TensorN_t m_weights;
		bias_t m_bias;

		TensorN_t m_grad_weights;
		bias_t m_grad_bias;

		aux_tensor_type m_im2col_aux_tensor;
		TensorN_t m_grad_input_aux_tensor;
		/// CNHW GEMM buffer for \ref im2ColConvolution (reused across forwards; avoids per-call alloc).
		TensorN_t m_gemm_cnhw_layout;
#if NEURAL_CUDNN_ENABLED
		/// cuDNN conv forward/backward workspace (single buffer sized to max need; not copied on \c ConvolutionalLayer copy).
		void *m_cudnn_workspace = nullptr;
		std::size_t m_cudnn_workspace_capacity = 0;
#endif
};

template <typename TensorN_t>
ConvolutionalLayer<TensorN_t>::ConvolutionalLayer( std::size_t out_channels, std::size_t kernel_size )
	: m_out_channels( out_channels )
	, m_kernel_size( kernel_size )
{
}

#if NEURAL_CUDNN_ENABLED
template < typename TensorN_t >
ConvolutionalLayer<TensorN_t>::~ConvolutionalLayer()
{
	if ( m_cudnn_workspace != nullptr ) {
		(void)cudaFree( m_cudnn_workspace );
		m_cudnn_workspace = nullptr;
		m_cudnn_workspace_capacity = 0;
	}
}

template < typename TensorN_t >
ConvolutionalLayer<TensorN_t>::ConvolutionalLayer( ConvolutionalLayer const &other )
	: LayerBase<TensorN_t>( other )
	, m_out_channels( other.m_out_channels )
	, m_kernel_size( other.m_kernel_size )
	, m_weights( other.m_weights )
	, m_bias( other.m_bias )
	, m_grad_weights( other.m_grad_weights )
	, m_grad_bias( other.m_grad_bias )
	, m_im2col_aux_tensor( other.m_im2col_aux_tensor )
	, m_grad_input_aux_tensor( other.m_grad_input_aux_tensor )
	, m_gemm_cnhw_layout( other.m_gemm_cnhw_layout )
	, m_cudnn_workspace( nullptr )
	, m_cudnn_workspace_capacity( 0 )
{
}

template < typename TensorN_t >
ConvolutionalLayer<TensorN_t>::ConvolutionalLayer( ConvolutionalLayer &&other ) noexcept
	: LayerBase<TensorN_t>( std::move( other ) )
	, m_out_channels( other.m_out_channels )
	, m_kernel_size( other.m_kernel_size )
	, m_weights( std::move( other.m_weights ) )
	, m_bias( std::move( other.m_bias ) )
	, m_grad_weights( std::move( other.m_grad_weights ) )
	, m_grad_bias( std::move( other.m_grad_bias ) )
	, m_im2col_aux_tensor( std::move( other.m_im2col_aux_tensor ) )
	, m_grad_input_aux_tensor( std::move( other.m_grad_input_aux_tensor ) )
	, m_gemm_cnhw_layout( std::move( other.m_gemm_cnhw_layout ) )
	, m_cudnn_workspace( other.m_cudnn_workspace )
	, m_cudnn_workspace_capacity( other.m_cudnn_workspace_capacity )
{
	other.m_cudnn_workspace = nullptr;
	other.m_cudnn_workspace_capacity = 0;
}

template < typename TensorN_t >
ConvolutionalLayer<TensorN_t> &ConvolutionalLayer<TensorN_t>::operator=( ConvolutionalLayer const &other )
{
	if ( this != &other ) {
		if ( m_cudnn_workspace != nullptr ) {
			(void)cudaFree( m_cudnn_workspace );
			m_cudnn_workspace = nullptr;
			m_cudnn_workspace_capacity = 0;
		}
		LayerBase<TensorN_t>::operator=( other );
		m_out_channels = other.m_out_channels;
		m_kernel_size = other.m_kernel_size;
		m_weights = other.m_weights;
		m_bias = other.m_bias;
		m_grad_weights = other.m_grad_weights;
		m_grad_bias = other.m_grad_bias;
		m_im2col_aux_tensor = other.m_im2col_aux_tensor;
		m_grad_input_aux_tensor = other.m_grad_input_aux_tensor;
		m_gemm_cnhw_layout = other.m_gemm_cnhw_layout;
	}
	return *this;
}

template < typename TensorN_t >
ConvolutionalLayer<TensorN_t> &ConvolutionalLayer<TensorN_t>::operator=( ConvolutionalLayer &&other ) noexcept
{
	if ( this != &other ) {
		if ( m_cudnn_workspace != nullptr ) {
			(void)cudaFree( m_cudnn_workspace );
		}
		LayerBase<TensorN_t>::operator=( std::move( other ) );
		m_out_channels = other.m_out_channels;
		m_kernel_size = other.m_kernel_size;
		m_weights = std::move( other.m_weights );
		m_bias = std::move( other.m_bias );
		m_grad_weights = std::move( other.m_grad_weights );
		m_grad_bias = std::move( other.m_grad_bias );
		m_im2col_aux_tensor = std::move( other.m_im2col_aux_tensor );
		m_grad_input_aux_tensor = std::move( other.m_grad_input_aux_tensor );
		m_gemm_cnhw_layout = std::move( other.m_gemm_cnhw_layout );
		m_cudnn_workspace = other.m_cudnn_workspace;
		m_cudnn_workspace_capacity = other.m_cudnn_workspace_capacity;
		other.m_cudnn_workspace = nullptr;
		other.m_cudnn_workspace_capacity = 0;
	}
	return *this;
}
#endif

template < typename TensorN_t >
void ConvolutionalLayer< TensorN_t >::ensure_weights( std::size_t in_channels )
{
	if ( m_weights.size() > 0 ) {
		if ( m_weights.shape()[1] != in_channels ) {
			throw std::logic_error(
			    "ConvolutionalLayer: input channel count does not match allocated weights" );
		}
		return;
	}
	std::array<std::size_t, 4> const wshape{ m_out_channels, in_channels, m_kernel_size,
	                                          m_kernel_size };
	m_weights      = TensorN_t( wshape );
	m_bias         = bias_t( { 1, m_out_channels, 1, 1 } );
	m_grad_weights = TensorN_t( wshape );
	m_grad_bias    = bias_t( { 1, m_out_channels, 1, 1 } );
	auto const &ws = m_weights.shape();
	std::size_t const fan_in = ws[1] * ws[2] * ws[3];
	std::mt19937_64 wgen( std::random_device{}() );
	std::uint64_t const s = wgen();
	// He (uniform): U(-√(2/fan_in), √(2/fan_in)); bias left zero (constructors zero-init).
	m_weights.randomizeHe( fan_in, s );
}

template < typename TensorN_t >
void ConvolutionalLayer< TensorN_t >::initialize()
{
	if ( m_weights.size() == 0 ) {
		return;
	}
	auto const &ws = m_weights.shape();
	std::size_t const fan_in = ws[1] * ws[2] * ws[3];
	std::mt19937_64 wgen( std::random_device{}() );
	std::uint64_t const s = wgen();
	// He (uniform): U(-√(2/fan_in), √(2/fan_in)); bias left zero (constructors zero-init).
	m_weights.randomizeHe( fan_in, s );
}

template < typename TensorN_t >
TensorN_t *ConvolutionalLayer< TensorN_t >::forward()
{
	TensorN_t *input = this->getInput();
	ensure_weights( input->shape()[1] );
	TensorN_t *output = this->getOutput();
#if NEURAL_CUDNN_ENABLED
	if constexpr ( is_cuda_tensor4_v<TensorN_t> ) {
		input->im2ColConvolution( m_weights, m_im2col_aux_tensor, *output, m_gemm_cnhw_layout,
		                          &m_cudnn_workspace, &m_cudnn_workspace_capacity );
	} else
#endif
	{
		input->im2ColConvolution( m_weights, m_im2col_aux_tensor, *output, m_gemm_cnhw_layout );
	}

	output->addColorChannelInPlace( m_bias );

#if NEURAL_CUDA_ENABLED
	if constexpr ( is_cuda_tensor4_v<TensorN_t> ) {
		cuda_layer_sync();
	}
#endif
	return output;
}

template < typename TensorN_t >
TensorN_t *ConvolutionalLayer< TensorN_t >::backward()
{
	if ( m_weights.size() == 0 ) {
		throw std::logic_error( "ConvolutionalLayer::backward: forward has not been run" );
	}
#if NEURAL_CUDNN_ENABLED
	if constexpr ( is_cuda_tensor4_v<TensorN_t> ) {
		convolutional_backward_cudnn( *this->getGradInput(), m_weights,
		                              *this->getInput(), m_grad_weights,
		                              m_grad_bias, *this->getGradOutput(),
		                              &m_cudnn_workspace, &m_cudnn_workspace_capacity );
	} else
#endif
	{
		convolutional_backward_im2col(
		    *this->getGradInput(), m_weights, m_im2col_aux_tensor,
		    this->getInput()->shape(), m_grad_weights, m_grad_bias,
		    *this->getGradOutput(), m_grad_input_aux_tensor );
	}
#if NEURAL_CUDA_ENABLED
	if constexpr ( is_cuda_tensor4_v<TensorN_t> ) {
		cuda_layer_sync();
	}
#endif
	return this->getGradOutput();
}

} // namespace neural

#endif
