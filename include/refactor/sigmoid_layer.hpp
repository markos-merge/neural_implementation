#ifndef REFACTOR_SIGMOID_LAYER_HPP
#define REFACTOR_SIGMOID_LAYER_HPP

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
class SigmoidLayer : public LayerBase<T, Device> {
public:
	void forward();
	void backward();

private:
	using tensor_t = std::conditional_t<std::is_same_v< Device, Cuda >, CudaTensor<T>, Tensor<T>>;
};

//Implementation
//Cpu specialization
template <typename T, typename Device>
void SigmoidLayer<T, Device>::forward()
{
	T *input = this->getInput()->fwdData();
	std::vector< std::size_t > const &input_shape = this->getInput()->shape();
	std::size_t const rows = input_shape[0];
	std::size_t const cols = std::accumulate( input_shape.begin() + 1, input_shape.end(),
		std::size_t{1}, std::multiplies<std::size_t>() );

	if ( this->getOutput()->shape() != input_shape ) {
		this->getOutput()->resize( input_shape );
	}

	tensor_t in_view( input, rows, cols, false );
	tensor_t out_view( this->getOutput()->fwdData(), rows, cols, false );

	in_view.cwiseSigmoid( out_view );
#if NEURAL_CUDA_ENABLED
	if constexpr ( std::is_same_v<Device, Cuda> ) {
		cuda_layer_sync();
	}
#endif
}

template <typename T, typename Device>
void SigmoidLayer<T, Device>::backward()
{
	std::vector<std::size_t> const &g_in_shape = this->getGradInput()->shape();
	std::size_t const rows = g_in_shape[0];
	std::size_t const cols = std::accumulate( g_in_shape.begin() + 1, g_in_shape.end(),
		std::size_t{1}, std::multiplies<std::size_t>() );

	if ( this->getGradOutput()->shape() != g_in_shape ) {
		this->getGradOutput()->resize( g_in_shape );
	}

	tensor_t y( this->getOutput()->fwdData(), rows, cols, false );
	tensor_t grad_in_view( this->getGradInput()->fwdData(), rows, cols, false );
	tensor_t grad_out_view( this->getGradOutput()->fwdData(), rows, cols, false );

//this needs a refactor for cuda tensor
	tensor_t const chain = y * y.cwiseOneMinus();
	grad_in_view.elementwiseMultiply( chain, grad_out_view );

#if NEURAL_CUDA_ENABLED
	if constexpr ( std::is_same_v<Device, Cuda> ) {
		cuda_layer_sync();
	}
#endif
}
} // namespace refactor
} // namespace neural

#endif // REFACTOR_SIGMOID_LAYER_HPP
