#ifndef CONVOLUTIONAL_LAYER_HPP
#define CONVOLUTIONAL_LAYER_HPP

#include <cstddef>
#include <array>
#include <stdexcept>
#include "layer_base.hpp"
#include "tensor_n.hpp"
#if NEURAL_CUDNN_ENABLED
#include "cuda_tensor_n.hpp"
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
};

template <typename TensorN_t>
ConvolutionalLayer<TensorN_t>::ConvolutionalLayer( std::size_t out_channels, std::size_t kernel_size )
	: m_out_channels( out_channels )
	, m_kernel_size( kernel_size )
{
}

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
	m_weights.randomizeHe( fan_in );
	m_bias = bias_t( m_bias.shape(), 0. );
}

template < typename TensorN_t >
void ConvolutionalLayer< TensorN_t >::initialize()
{
	if ( m_weights.size() == 0 ) {
		return;
	}
	auto const &ws = m_weights.shape();
	std::size_t const fan_in = ws[1] * ws[2] * ws[3];
	m_weights.randomizeHe( fan_in );
	m_bias = bias_t( m_bias.shape(), 0. );
}

template < typename TensorN_t >
TensorN_t *ConvolutionalLayer< TensorN_t >::forward()
{
	TensorN_t *input = this->getInput();
	ensure_weights( input->shape()[1] );
	TensorN_t *output = this->getOutput();
	input->im2ColConvolution( m_weights, m_im2col_aux_tensor, *output, m_gemm_cnhw_layout );

	output->addColorChannelInPlace( m_bias );

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
		convolutional_backward_cudnn( *this->getGradInput(), m_weights, *this->getInput(), m_grad_weights,
		                              m_grad_bias, *this->getGradOutput() );
	} else
#endif
	{
		convolutional_backward_im2col( *this->getGradInput(), m_weights, m_im2col_aux_tensor,
		                               this->getInput()->shape(), m_grad_weights, m_grad_bias,
		                               *this->getGradOutput(), m_grad_input_aux_tensor );
	}
	return this->getGradOutput();
}

} // namespace neural

#endif
