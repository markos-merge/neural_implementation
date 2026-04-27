#ifndef SIGMOID_LAYER_HPP
#define SIGMOID_LAYER_HPP
#include "layer_base.hpp"
#include "tensor_like.hpp"
#if NEURAL_CUDA_ENABLED
#include "cuda_tensor.hpp"
#include "neural_cuda_layer_sync.hpp"
#endif

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
#if NEURAL_CUDA_ENABLED
	if constexpr ( is_cuda_tensor_v<Tensor> ) {
		cuda_layer_sync();
	}
#endif
	return this->getOutput();
}

template <typename Tensor >
Tensor *SigmoidLayer<Tensor>::backward()
{
	*this->getGradOutput() =
	    ( *this->getGradInput() ) * m_output * m_output.cwiseOneMinus();
#if NEURAL_CUDA_ENABLED
	if constexpr ( is_cuda_tensor_v<Tensor> ) {
		cuda_layer_sync();
	}
#endif
	return this->getGradOutput();
}

} // namespace neural

#endif
