#ifndef SIGMOID_LAYER_HPP
#define SIGMOID_LAYER_HPP
#include "tensor_like.hpp"

namespace neural {

template <typename Tensor>
class SigmoidLayer
{
	static_assert( TensorLike<Tensor>, "Tensor must be a TensorLike type" );
	public:
		SigmoidLayer();

		Tensor forward( Tensor const &input );
		Tensor backward( Tensor const &grad_output );
	private:
	// Cache for backward pass
		Tensor m_output;
};

template <typename Tensor >
SigmoidLayer<Tensor>::SigmoidLayer()
{
}

template <typename Tensor >
Tensor SigmoidLayer<Tensor>::forward( Tensor const &input )
{
	m_output = input.cwiseSigmoid();

	return m_output;
}

template <typename Tensor >
Tensor SigmoidLayer<Tensor>::backward( Tensor const &grad_output )
{
	return grad_output * m_output * m_output.cwiseOneMinus();
}

}

#endif