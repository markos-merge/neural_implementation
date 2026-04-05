#ifndef RELU_LAYER_HPP
#define RELU_LAYER_HPP
#include "layer_base.hpp"

namespace neural {

template <typename Tensor>
class ReLULayer : public LayerBase< Tensor >
{
	public:
		using tensor_t = Tensor;
	public:
		ReLULayer();

		Tensor *forward();
		Tensor *backward();

	private:
		Tensor m_mask;
};

template <typename Tensor >
ReLULayer<Tensor>::ReLULayer() = default;

template <typename Tensor >
Tensor *ReLULayer<Tensor>::forward()
{
	Tensor *input = this->getInput();
	if ( m_mask.shape() != input->shape() ) {
		m_mask = Tensor( input->shape() );
	}

	m_mask.cwiseGreaterInPlace( *input, static_cast<typename Tensor::value_type>(0.) );
	( *this->getOutput() ) = *input;
	*this->getOutput() *= m_mask;

	return this->getOutput();
}

template <typename Tensor >
Tensor *ReLULayer<Tensor>::backward()
{
	(*this->getGradOutput() ) = (*this->getGradInput() );
	*this->getGradOutput() *= m_mask;

	return this->getGradOutput();
}

}

#endif
