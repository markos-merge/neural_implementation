#ifndef FLATTEN_LAYER_HPP
#define FLATTEN_LAYER_HPP

#include <array>
#include <cstddef>
#include <cstring>

namespace neural {


template <typename TensorN_t, typename Tensor2D_t>
class FlattenLayer
{
	static_assert( TensorN_t::rank_v == 4, "FlattenLayer requires a rank-4 TensorN" );

	public:
		void setInputOutputTensors( TensorN_t *input, Tensor2D_t *output );
		void setGradInputOutputTensors( Tensor2D_t *grad_input, TensorN_t *grad_output );

		Tensor2D_t *forward();
		TensorN_t  *backward();

	private:
		TensorN_t  *m_input       = nullptr;
		Tensor2D_t *m_output      = nullptr;
		Tensor2D_t *m_grad_input  = nullptr;
		TensorN_t  *m_grad_output = nullptr;

		std::array<std::size_t, 4> m_original_shape{};
};

template <typename TensorN_t, typename Tensor2D_t>
void FlattenLayer<TensorN_t, Tensor2D_t>::setInputOutputTensors( TensorN_t *input,
                                                                  Tensor2D_t *output )
{
	m_input  = input;
	m_output = output;
}

template <typename TensorN_t, typename Tensor2D_t>
void FlattenLayer<TensorN_t, Tensor2D_t>::setGradInputOutputTensors( Tensor2D_t *grad_input,
                                                                      TensorN_t  *grad_output )
{
	m_grad_input  = grad_input;
	m_grad_output = grad_output;
}

template <typename TensorN_t, typename Tensor2D_t>
Tensor2D_t *FlattenLayer<TensorN_t, Tensor2D_t>::forward()
{
	m_original_shape = m_input->shape();

	std::size_t const N    = m_original_shape[0];
	std::size_t const flat = m_original_shape[1] * m_original_shape[2] * m_original_shape[3];

	if ( m_output->rows() != N || m_output->cols() != flat ) {
		*m_output = Tensor2D_t( N, flat );
	}

	std::memcpy( m_output->data(), m_input->data(),
	             m_input->size() * sizeof( typename TensorN_t::value_type ) );

	return m_output;
}

template <typename TensorN_t, typename Tensor2D_t>
TensorN_t *FlattenLayer<TensorN_t, Tensor2D_t>::backward()
{
	if ( m_grad_output->shape() != m_original_shape ) {
		*m_grad_output = TensorN_t( m_original_shape );
	}

	std::memcpy( m_grad_output->data(), m_grad_input->data(),
	             m_grad_input->size() * sizeof( typename Tensor2D_t::value_type ) );

	return m_grad_output;
}

} // namespace neural

#endif // FLATTEN_LAYER_HPP
