#ifndef RELU_LAYER_HPP
#define RELU_LAYER_HPP
#include "tensor_like.hpp"

namespace neural {

template <typename Tensor>
class ReLULayer
{
	static_assert( TensorLike<Tensor>, "Tensor must be a TensorLike type" );
	public:
		ReLULayer();
		Tensor forward( Tensor const &input );
		Tensor backward( Tensor const &grad_output );
	private:
	// Cache for backward pass
		Tensor m_mask;
};

template <typename Tensor >
ReLULayer<Tensor>::ReLULayer()
{
}

template <typename Tensor >
Tensor ReLULayer<Tensor>::forward( Tensor const &input )
{
	m_mask = input.cwiseGreater( static_cast<typename Tensor::value_type>(0.) );

	return input*m_mask;
}

template <typename Tensor >
Tensor ReLULayer<Tensor>::backward( Tensor const &grad_output )
{
	return grad_output*m_mask;
}

}

#endif