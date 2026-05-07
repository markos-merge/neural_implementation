#ifndef MAX_POOL_LAYER_HPP
#define MAX_POOL_LAYER_HPP

#include <cstddef>
#include <array>
#include "layer_base.hpp"
#include "tensor_n.hpp"
#if NEURAL_CUDA_ENABLED
#include "cuda_tensor_n.hpp"
#include "neural_cuda_layer_sync.hpp"
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
		this->getOutput()->throw_if_non_owning_realloc(
		    "MaxPoolLayer::forward: cannot reallocate non-owning output" );
		TensorN_t const fresh( output_shape );
		*this->getOutput() = fresh;
	}

#if NEURAL_CUDNN_ENABLED
	if constexpr ( is_cuda_tensor4_v<TensorN_t> ) {
		max_pool_forward_cudnn( *this->getInput(), *this->getOutput(), m_argmax, m_pool_size, m_stride );
	} else
#endif
	{
		max_pool_forward_nchw( *this->getInput(), m_argmax, *this->getOutput(), m_pool_size, m_stride );
	}

#if NEURAL_CUDA_ENABLED
	if constexpr ( is_cuda_tensor4_v<TensorN_t> ) {
		cuda_layer_sync();
	}
#endif
	return this->getOutput();
}

template <typename TensorN_t>
TensorN_t *MaxPoolLayer<TensorN_t>::backward()
{
	if ( this->getGradOutput()->shape() != this->getInput()->shape() ) {
		this->getGradOutput()->throw_if_non_owning_realloc(
		    "MaxPoolLayer::backward: cannot reallocate non-owning grad output" );
		TensorN_t const fresh( this->getInput()->shape() );
		*this->getGradOutput() = fresh;
	}

#if NEURAL_CUDNN_ENABLED
	if constexpr ( is_cuda_tensor4_v<TensorN_t> ) {
		max_pool_backward_cudnn( *this->getInput(), *this->getOutput(), *this->getGradInput(),
		                         *this->getGradOutput(), m_argmax, m_pool_size, m_stride );
	} else
#endif
	{
		max_pool_backward_nchw( *this->getGradInput(), m_argmax, *this->getGradOutput(), m_pool_size,
		                        m_stride );
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
