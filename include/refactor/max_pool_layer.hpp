#ifndef REFACTOR_MAX_POOL_LAYER_HPP
#define REFACTOR_MAX_POOL_LAYER_HPP

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "device.hpp"
#include "layer_base.hpp"
#include "tensor_n.hpp"
#include "cuda_tensor_n.hpp"
#include "neural_cuda_layer_sync.hpp"

namespace neural {
namespace refactor {

template <typename T, typename Device = Cpu>
class MaxPoolLayer : public LayerBase<T, Device>
{
public:
	MaxPoolLayer( std::size_t pool_size, std::size_t stride );

	std::size_t poolSize() const;
	std::size_t stride() const;

	void forward();
	void backward();

private:
	using tensor_t =
	    std::conditional_t<std::is_same_v<Device, Cuda>, CudaTensor4<T>, TensorN<4, T>>;
	using argmax_tensor_t = typename tensor_t::template tensor_alias<4, std::size_t>;

	std::size_t m_pool_size;
	std::size_t m_stride;

	argmax_tensor_t m_argmax;
};

// -----------------------------------------------------------------------------
// Implementation
// -----------------------------------------------------------------------------

template <typename T, typename Device>
MaxPoolLayer<T, Device>::MaxPoolLayer( std::size_t pool_size, std::size_t stride )
	: m_pool_size( pool_size )
	, m_stride( stride )
{
	if ( m_pool_size == 0 || m_stride == 0 ) {
		throw std::invalid_argument( "MaxPoolLayer: pool_size and stride must be positive" );
	}
}

template <typename T, typename Device>
std::size_t MaxPoolLayer<T, Device>::poolSize() const
{
	return m_pool_size;
}

template <typename T, typename Device>
std::size_t MaxPoolLayer<T, Device>::stride() const
{
	return m_stride;
}

template <typename T, typename Device>
void MaxPoolLayer<T, Device>::forward()
{
	std::vector<std::size_t> const &in_vec = this->getInput()->shape();
	if ( in_vec.size() != 4 ) {
		throw std::invalid_argument( "MaxPoolLayer::forward: input latch must have rank 4 (NCHW)" );
	}
	std::array<std::size_t, 4> const in_shape = nn_shape_vec_to_fixed<4>( in_vec );
	if ( in_shape[2] < m_pool_size || in_shape[3] < m_pool_size ) {
		throw std::invalid_argument(
		    "MaxPoolLayer::forward: spatial dimensions must be >= pool_size" );
	}

	std::array<std::size_t, 4> out_shape = in_shape;
	out_shape[2] = ( in_shape[2] - m_pool_size ) / m_stride + 1;
	out_shape[3] = ( in_shape[3] - m_pool_size ) / m_stride + 1;

	std::vector<std::size_t> const out_vec( out_shape.begin(), out_shape.end() );
	if ( this->getOutput()->shape() != out_vec ) {
		this->getOutput()->resize( out_vec );
	}

	if ( m_argmax.shape() != out_shape ) {
		m_argmax = argmax_tensor_t( out_shape );
	}

	tensor_t const in_view( this->getInput()->fwdData(), in_vec, false );
	tensor_t out_view( this->getOutput()->fwdData(), out_vec, false );

	if constexpr ( std::is_same_v<Device, Cpu> ) {
		max_pool_forward_nchw( in_view, m_argmax, out_view, m_pool_size, m_stride );
	} else {
		max_pool_forward_cudnn( in_view, out_view, m_argmax, m_pool_size, m_stride );
		cuda_layer_sync();
	}
}

template <typename T, typename Device>
void MaxPoolLayer<T, Device>::backward()
{
	std::vector<std::size_t> const &in_vec = this->getInput()->shape();
	if ( in_vec.size() != 4 ) {
		throw std::invalid_argument( "MaxPoolLayer::backward: input latch must have rank 4 (NCHW)" );
	}

	if ( this->getGradOutput()->shape() != in_vec ) {
		this->getGradOutput()->resize( in_vec );
	}

	std::vector<std::size_t> const out_vec = this->getOutput()->shape();
	if ( out_vec.size() != 4 ) {
		throw std::logic_error( "MaxPoolLayer::backward: output latch rank must be 4" );
	}
	std::vector<std::size_t> const grad_upstream_vec = this->getGradInput()->shape();
	if ( grad_upstream_vec.size() != 4 || grad_upstream_vec != out_vec ) {
		throw std::logic_error(
		    "MaxPoolLayer::backward: upstream gradient latch shape must match forward output" );
	}

	tensor_t const in_view( this->getInput()->fwdData(), in_vec, false );
	tensor_t const out_view( this->getOutput()->fwdData(), out_vec, false );
	tensor_t const g_up( this->getGradInput()->fwdData(), grad_upstream_vec, false );
	tensor_t g_in( this->getGradOutput()->fwdData(), in_vec, false );

	if constexpr ( std::is_same_v<Device, Cpu> ) {
		max_pool_backward_nchw( g_up, m_argmax, g_in, m_pool_size, m_stride );
	} else {
		max_pool_backward_cudnn( in_view, out_view, g_up, g_in, m_argmax, m_pool_size, m_stride );
		cuda_layer_sync();
	}
}

} // namespace refactor
} // namespace neural

#endif // REFACTOR_MAX_POOL_LAYER_HPP
