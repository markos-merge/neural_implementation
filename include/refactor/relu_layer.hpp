#ifndef REFACTOR_RELU_LAYER_HPP
#define REFACTOR_RELU_LAYER_HPP

#include "layer_base.hpp"
#include "latch_layer.hpp"
#include "tensor.hpp"
#include <array>
#include <numeric>
#include <type_traits>
#if NEURAL_CUDA_ENABLED
#include "cuda_tensor.hpp"
#endif

namespace neural {
namespace refactor {

template <typename T, typename Device = Cpu>
class ReLULayer : public LayerBase<T, Device> {
public:
	void forward();
	void backward();

private:
	using tensor_t = std::conditional_t<std::is_same_v< Device, Cuda >, CudaTensor<T>, Tensor<T>>;
private:
	tensor_t m_mask;
};

//Implementation
//Cpu specialization
template <typename T, typename Device>
void ReLULayer<T, Device>::forward()
{
	T *input = this->getInput()->fwdData();
	std::vector< std::size_t > const &input_shape = this->getInput()->shape();
	std::size_t const rows = input_shape[0];
	std::size_t const cols = std::accumulate( input_shape.begin() + 1, input_shape.end(),
		std::size_t{1}, std::multiplies<std::size_t>() );
	std::array<std::size_t, 2> const mask_shape = m_mask.shape();
	if ( mask_shape[0] != rows || mask_shape[1] != cols ) {
		m_mask = tensor_t( rows, cols );
	}

	if ( this->getOutput()->shape() != input_shape ) {
		this->getOutput()->resize( input_shape );
	}

	tensor_t in_view( input, rows, cols, false );
	tensor_t out_view( this->getOutput()->fwdData(), rows, cols, false );

	m_mask.cwiseGreaterInPlace( in_view, static_cast<T>(0.) );

	in_view.elementwiseMultiply( m_mask, out_view );
}

template <typename T, typename Device>
void ReLULayer<T, Device>::backward()
{
	T *grad_input = this->getGradInput()->fwdData();
	std::vector< std::size_t > const &grad_input_shape = this->getGradInput()->shape();
	std::size_t const rows = grad_input_shape[0];
	std::size_t const cols = std::accumulate( grad_input_shape.begin() + 1, grad_input_shape.end(),
		std::size_t{1}, std::multiplies<std::size_t>() );
	std::array<std::size_t, 2> const mask_shape = m_mask.shape();
	if ( mask_shape[0] != rows || mask_shape[1] != cols ) {
		m_mask = tensor_t( rows, cols );
	}

	if( this->getGradOutput()->shape() != grad_input_shape ) {
		this->getGradOutput()->resize( grad_input_shape );
	}

	tensor_t grad_input_view( grad_input, rows, cols, false );
	tensor_t grad_output_view( this->getGradOutput()->fwdData(), rows, cols, false );

	grad_input_view.elementwiseMultiply( m_mask, grad_output_view );
}

} // namespace refactor
} // namespace neural

#endif // REFACTOR_RELU_LAYER_HPP
