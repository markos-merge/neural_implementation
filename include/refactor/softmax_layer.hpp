#ifndef REFACTOR_SOFTMAX_LAYER_HPP
#define REFACTOR_SOFTMAX_LAYER_HPP

#include "layer_base.hpp"
#include "latch_layer.hpp"
#include "tensor.hpp"
#include <numeric>
#include <type_traits>
#if NEURAL_CUDA_ENABLED
#include "cuda_tensor.hpp"
#include "neural_cuda_layer_sync.hpp"
#endif

namespace neural {
namespace refactor {

template <typename T, typename Device = Cpu>
class SoftmaxLayer : public LayerBase<T, Device>
{
public:
	void forward();
	void backward();

private:
	using tensor_t = std::conditional_t<std::is_same_v<Device, Cuda>, CudaTensor<T>, Tensor<T>>;
private:
	tensor_t m_probs;
};

template <typename T, typename Device>
void SoftmaxLayer<T, Device>::forward()
{
	T *input = this->getInput()->fwdData();
	std::vector<std::size_t> const &input_shape = this->getInput()->shape();
	std::size_t const rows = input_shape[0];
	std::size_t const cols =
	    std::accumulate( input_shape.begin() + 1, input_shape.end(), std::size_t{1},
	                     std::multiplies<std::size_t>() );

	if ( this->getOutput()->shape() != input_shape ) {
		this->getOutput()->resize( input_shape );
	}

	tensor_t input_view( input, rows, cols, false );

	if ( input_view.size() == 0 ) {
		return;
	}

	tensor_t const row_max = input_view.max_along_axis( 1 );
	tensor_t shifted = input_view.subtractColwise( row_max );
	tensor_t exp_shifted = shifted.cwiseExp();
	tensor_t const sum_exp = exp_shifted.sum_along_axis( 1 );
	m_probs = exp_shifted;
	m_probs.divideRowsWithColInPlace( sum_exp );

	tensor_t out_view( this->getOutput()->fwdData(), rows, cols, false );
	out_view = m_probs;

#if NEURAL_CUDA_ENABLED
	if constexpr ( std::is_same_v<Device, Cuda> ) {
		cuda_layer_sync();
	}
#endif
}

template <typename T, typename Device>
void SoftmaxLayer<T, Device>::backward()
{
	std::vector<std::size_t> const &gin_shape = this->getGradInput()->shape();
	std::size_t const rows = gin_shape[0];
	std::size_t const cols =
	    std::accumulate( gin_shape.begin() + 1, gin_shape.end(), std::size_t{1},
	                     std::multiplies<std::size_t>() );

	if ( this->getGradOutput()->shape() != gin_shape ) {
		this->getGradOutput()->resize( gin_shape );
	}

	tensor_t grad_view( this->getGradInput()->fwdData(), rows, cols, false );

	if ( grad_view.size() == 0 ) {
		return;
	}

	tensor_t const dot = ( grad_view * m_probs ).sum_along_axis( 1 );
	tensor_t const diff = grad_view.subtractColwise( dot );
	tensor_t const result = m_probs * diff;

	tensor_t grad_out_view( this->getGradOutput()->fwdData(), rows, cols, false );
	grad_out_view = result;

#if NEURAL_CUDA_ENABLED
	if constexpr ( std::is_same_v<Device, Cuda> ) {
		cuda_layer_sync();
	}
#endif
}

} // namespace refactor
} // namespace neural

#endif // REFACTOR_SOFTMAX_LAYER_HPP
