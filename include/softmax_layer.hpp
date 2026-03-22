#ifndef SOFTMAX_LAYER_HPP
#define SOFTMAX_LAYER_HPP
#include "tensor_like.hpp"

namespace neural {

template <typename Tensor>
class SoftmaxLayer
{
	public:
		SoftmaxLayer();
	
		Tensor forward( Tensor const &input );
		Tensor backward( Tensor const &grad_output );
	private:
		Tensor m_input;
		Tensor m_output;
};

template <typename Tensor >
SoftmaxLayer<Tensor>::SoftmaxLayer()
{
}

template <typename Tensor >
Tensor SoftmaxLayer<Tensor>::forward( Tensor const &input )
{
	m_input = input;
	m_output = m_input.cwiseExp();
	Tensor sum = m_output.sum_along_axis( 1 );
	m_output.divideRowsWithColInPlace( sum );
	
	return m_output;
}

template <typename Tensor >
Tensor SoftmaxLayer<Tensor>::backward( Tensor const &grad_output )
{
	Tensor const dot = ( grad_output * m_output ).sum_along_axis( 1 );
	Tensor const diff = grad_output.subtractColwise( dot );
	return m_output * diff;
}
}

#endif
