#ifndef REFACTOR_MSE_LOSS_HPP
#define REFACTOR_MSE_LOSS_HPP

#include <cstddef>
#include <type_traits>
#include "device.hpp"
#include "tensor.hpp"
#if NEURAL_CUDA_ENABLED
#include "cuda_tensor.hpp"
#endif

namespace neural {
namespace refactor {

// L = (1/n) * sum((pred - target)^2)
// dL/dpred = (2/n) * (pred - target)
template <typename T, typename Device = Cpu>
class MSELoss
{
public:
	using tensor_t = std::conditional_t<std::is_same_v<Device, Cuda>, CudaTensor<T>, Tensor<T>>;

	// pred: [N * C], target: [N * C], grad_out: [N * C] — caller allocates.
	// Returns scalar loss value.
	T operator()( T const *pred, T const *target, std::size_t N, std::size_t C,
	              T *grad_out ) const;

private:
	mutable tensor_t m_diff;
	mutable tensor_t m_grad;
};

template <typename T, typename Device>
T MSELoss<T, Device>::operator()( T const *pred, T const *target, std::size_t N, std::size_t C,
                                   T *grad_out ) const
{
	tensor_t const pred_view( const_cast<T *>( pred ), N, C, false );
	tensor_t const tgt_view( const_cast<T *>( target ), N, C, false );

	m_diff = pred_view - tgt_view;

	std::size_t const n = N * C;
	T const inv_n       = T( 1 ) / static_cast<T>( n );
	T const scale       = T( 2 ) * inv_n;
	T const loss        = ( m_diff * m_diff ).sum() * inv_n;

	m_grad = m_diff * scale;

	tensor_t grad_view( grad_out, N, C, false );
	grad_view = m_grad;

	return loss;
}

} // namespace refactor
} // namespace neural

#endif // REFACTOR_MSE_LOSS_HPP
