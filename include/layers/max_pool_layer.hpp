#ifndef MAX_POOL_LAYER_HPP
#define MAX_POOL_LAYER_HPP

#include <cstddef>
#include <array>
#include "layer_base.hpp"
#include "tensor_n.hpp"
#if NEURAL_CUDNN_ENABLED
#include "cuda_tensor_n.hpp"
#endif

namespace neural {

template <typename TensorN_t>
class MaxPoolLayer : public LayerBase<TensorN_t>
{
	public:
		using tensor_t = TensorN_t;
	public:
		MaxPoolLayer( std::size_t pool_size, std::size_t stride );

		std::size_t poolSize() const { return m_pool_size; }
		std::size_t stride() const { return m_stride; }

		TensorN_t *forward();
		TensorN_t *backward();

	private:
		using argmax_tensor_type = typename TensorN_t::template tensor_alias<
		    4, std::size_t>;

		std::size_t m_pool_size;
		std::size_t m_stride;

		argmax_tensor_type m_argmax;
};

template <typename TensorN_t>
MaxPoolLayer<TensorN_t>::MaxPoolLayer( std::size_t pool_size, std::size_t stride )
	: m_pool_size( pool_size )
	, m_stride( stride )
{
}

template <typename TensorN_t>
TensorN_t *MaxPoolLayer<TensorN_t>::forward()
{
	std::array< std::size_t, 4 > const in_shape = this->getInput()->shape();
	std::array< std::size_t, 4 > output_shape    = in_shape;
	output_shape[3] = ( in_shape[3] - m_pool_size ) / m_stride + 1;
	output_shape[2] = ( in_shape[2] - m_pool_size ) / m_stride + 1;
	if ( m_argmax.shape() != output_shape ) {
		m_argmax = argmax_tensor_type( output_shape );
	}

	if ( this->getOutput()->shape() != output_shape ) {
		*this->getOutput() = TensorN_t( output_shape );
	}

#if NEURAL_CUDNN_ENABLED
	if constexpr ( is_cuda_tensor4_v<TensorN_t> ) {
		max_pool_forward_cudnn( *this->getInput(), *this->getOutput(), m_argmax, m_pool_size, m_stride );
		return this->getOutput();
	} else
#endif
	{
		max_pool_forward_nchw( *this->getInput(), m_argmax, *this->getOutput(), m_pool_size, m_stride );
	}

	return this->getOutput();
}

template <typename TensorN_t>
TensorN_t *MaxPoolLayer<TensorN_t>::backward()
{
	if ( this->getGradOutput()->shape() != this->getInput()->shape() ) {
		*this->getGradOutput() = TensorN_t( this->getInput()->shape() );
	}

#if NEURAL_CUDNN_ENABLED
	if constexpr ( is_cuda_tensor4_v<TensorN_t> ) {
		max_pool_backward_cudnn( *this->getInput(), *this->getOutput(), *this->getGradInput(),
		                         *this->getGradOutput(), m_argmax, m_pool_size, m_stride );
		return this->getGradOutput();
	} else
#endif
	{
		max_pool_backward_nchw( *this->getGradInput(), m_argmax, *this->getGradOutput(), m_pool_size,
		                        m_stride );
	}

	return this->getGradOutput();
}

} // namespace neural

#endif
