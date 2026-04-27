#ifndef RELU_LAYER_HPP
#define RELU_LAYER_HPP
#include "layer_base.hpp"
#if NEURAL_CUDA_ENABLED
#include "cuda_tensor.hpp"
#include "neural_cuda_layer_sync.hpp"
#endif

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
	this->getInput()->elementwiseMultiply( m_mask, *this->getOutput() );

#if NEURAL_CUDA_ENABLED
	if constexpr ( is_cuda_tensor_v<Tensor> ) {
		cuda_layer_sync();
	}
#endif
	return this->getOutput();
}

template <typename Tensor >
Tensor *ReLULayer<Tensor>::backward()
{
	this->getGradInput()->elementwiseMultiply( m_mask,*this->getGradOutput() );

#if NEURAL_CUDA_ENABLED
	if constexpr ( is_cuda_tensor_v<Tensor> ) {
		cuda_layer_sync();
	}
#endif
	return this->getGradOutput();
}

}

#endif
