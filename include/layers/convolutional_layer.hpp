#ifndef CONVOLUTIONAL_LAYER_HPP
#define CONVOLUTIONAL_LAYER_HPP

#include <cstddef>
#include <array>
#include "layer_base.hpp"

namespace neural {

template <typename TensorN_t>
class ConvolutionalLayer : public LayerBase< TensorN_t >
{
	public:
		using tensor_t = TensorN_t;
		using bias_t = TensorN_t;

		ConvolutionalLayer( std::size_t in_channels, std::size_t out_channels, std::size_t kernel_size );
	
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
		using aux_tensor_type = typename TensorN_t::template tensor_alias<
		    2, typename TensorN_t::value_type>;
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
ConvolutionalLayer<TensorN_t>::ConvolutionalLayer( std::size_t in_channels, std::size_t out_channels, std::size_t kernel_size )
	: m_weights( { out_channels, in_channels, kernel_size, kernel_size } )
	, m_bias( { 1, out_channels, 1, 1 } )
	, m_grad_weights( { out_channels, in_channels, kernel_size, kernel_size } )
	, m_grad_bias( { 1, out_channels, 1, 1 } )
{
}

template < typename TensorN_t >
void ConvolutionalLayer< TensorN_t >::initialize()
{
	auto const &ws = m_weights.shape();
	// He: fan_in = C_in * K_h * K_w (not just C_in).
	std::size_t const fan_in = ws[1] * ws[2] * ws[3];
	m_weights.randomizeHe( fan_in );
	m_bias = bias_t( m_bias.shape(), 0. );
}

template < typename TensorN_t >
TensorN_t *ConvolutionalLayer< TensorN_t >::forward()
{
	TensorN_t *output = this->getOutput();
	TensorN_t *input = this->getInput();
	input->im2ColConvolution( m_weights, m_im2col_aux_tensor, *output, m_gemm_cnhw_layout );

	output->addColorChannelInPlace( m_bias );

	return output;
}

template < typename TensorN_t >
TensorN_t *ConvolutionalLayer< TensorN_t >::backward()
{
	TensorN_t *grad_input = this->getGradInput();
	TensorN_t *grad_output = this->getGradOutput();

	auto const input_shape = this->getInput()->shape();
	auto const w_shape = m_weights.shape();

	grad_input->swapAxes( { 1, 0, 2, 3 }, m_grad_input_aux_tensor );

	m_grad_input_aux_tensor.multiply( m_im2col_aux_tensor, true, m_grad_weights );

	m_grad_input_aux_tensor.reduceSumToDim( 0, m_grad_bias );

	auto const grad_shape = m_grad_input_aux_tensor.shape();
	m_im2col_aux_tensor = aux_tensor_type(
		{ grad_shape[0], grad_shape[1] * grad_shape[2] * grad_shape[3] },
		m_grad_input_aux_tensor.data(),
		m_grad_input_aux_tensor.data() + m_grad_input_aux_tensor.size() );

	m_weights.swapAxes( { 1, 2, 3, 0 }, m_grad_input_aux_tensor );
	m_grad_input_aux_tensor.reshape( { w_shape[1] * w_shape[2] * w_shape[3], w_shape[0], 1, 1 } );

	TensorN_t grad_col( { w_shape[1] * w_shape[2] * w_shape[3], m_im2col_aux_tensor.shape()[1], 1, 1 } );
	m_grad_input_aux_tensor.multiply( m_im2col_aux_tensor, false, grad_col );

	grad_col.col2Im( w_shape, input_shape, *grad_output );

	return grad_output;
}

} // namespace neural

#endif