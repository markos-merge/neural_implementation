#ifndef SOFTMAX_LAYER_HPP
#define SOFTMAX_LAYER_HPP
#include "layer_base.hpp"
#include "tensor_like.hpp"
#if NEURAL_CUDA_ENABLED
#include "cuda_tensor.hpp"
#include "neural_cuda_layer_sync.hpp"
#endif

namespace neural {

template <typename Tensor>
class SoftmaxLayer : public LayerBase< Tensor >
{
	static_assert( TensorLike<Tensor>, "Tensor must be a TensorLike type" );
	public:
		SoftmaxLayer() = default;

		Tensor *forward();
		Tensor *backward();

	private:
		Tensor m_probs;
};

template <typename Tensor>
Tensor *SoftmaxLayer<Tensor>::forward()
{
	Tensor const &input = *this->getInput();
	Tensor const row_max = input.max_along_axis( 1 );
	Tensor shifted = input.subtractColwise( row_max );
	Tensor exp_shifted = shifted.cwiseExp();
	Tensor const sum_exp = exp_shifted.sum_along_axis( 1 );
	m_probs = exp_shifted;
	m_probs.divideRowsWithColInPlace( sum_exp );
	*this->getOutput() = m_probs;
#if NEURAL_CUDA_ENABLED
	if constexpr ( is_cuda_tensor_v<Tensor> ) {
		cuda_layer_sync();
	}
#endif
	return this->getOutput();
}

template <typename Tensor>
Tensor *SoftmaxLayer<Tensor>::backward()
{
	Tensor const &grad = *this->getGradInput();
	Tensor const dot = ( grad * m_probs ).sum_along_axis( 1 );
	Tensor const diff = grad.subtractColwise( dot );
	*this->getGradOutput() = m_probs * diff;
#if NEURAL_CUDA_ENABLED
	if constexpr ( is_cuda_tensor_v<Tensor> ) {
		cuda_layer_sync();
	}
#endif
	return this->getGradOutput();
}

} // namespace neural

#endif
