#ifndef SIGMOID_LAYER_HPP
#define SIGMOID_LAYER_HPP
#include "tensor_like.hpp"
#include "layer_base.hpp"

namespace neural {

template <typename Tensor>
class SigmoidLayer : public LayerBase< Tensor >
{
	static_assert( TensorLike<Tensor>, "Tensor must be a TensorLike type" );
	public:
		SigmoidLayer();

		Tensor *forward();
		Tensor *backward();

	private:
		Tensor m_output;
};

template <typename Tensor >
SigmoidLayer<Tensor>::SigmoidLayer() = default;

template <typename Tensor >
Tensor *SigmoidLayer<Tensor>::forward()
{
	m_output = this->getInput()->cwiseSigmoid();
	*this->getOutput() = m_output;
	return this->getOutput();
}

template <typename Tensor >
Tensor *SigmoidLayer<Tensor>::backward()
{
	*this->getGradOutput() =
	    ( *this->getGradInput() ) * m_output * m_output.cwiseOneMinus();
	return this->getGradOutput();
}

} // namespace neural

#endif
