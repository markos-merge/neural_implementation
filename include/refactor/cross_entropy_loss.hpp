#ifndef REFACTOR_CROSS_ENTROPY_LOSS_HPP
#define REFACTOR_CROSS_ENTROPY_LOSS_HPP

#include <cstddef>
#include <type_traits>
#include "device.hpp"
#include "tensor.hpp"
#if NEURAL_CUDA_ENABLED
#include "cuda_tensor.hpp"
#endif

namespace neural {
namespace refactor {

// Fused softmax + categorical cross-entropy on logits.
// L = -(1/N) * sum_n sum_c  target[n,c] * log(softmax(logit[n,c]))
// dL/dlogit[n,c] = (1/N) * (softmax[n,c] - target[n,c])
template <typename T, typename Device = Cpu>
class CrossEntropyLoss
{
public:
	using tensor_t = std::conditional_t<std::is_same_v<Device, Cuda>, CudaTensor<T>, Tensor<T>>;

	// logits: [N * C], target: [N * C] (one-hot or soft labels), grad_out: [N * C]
	// Returns scalar loss value.
	T operator()( T const *logits, T const *target, std::size_t N, std::size_t C,
	              T *grad_out ) const;

private:
	mutable tensor_t m_probs;
	mutable tensor_t m_grad;
	mutable tensor_t m_sum_exp;
};

template <typename T, typename Device>
T CrossEntropyLoss<T, Device>::operator()( T const *logits, T const *target, std::size_t N,
                                            std::size_t C, T *grad_out ) const
{
	tensor_t const logit_view( const_cast<T *>( logits ), N, C, false );
	tensor_t const tgt_view( const_cast<T *>( target ), N, C, false );

	tensor_t const row_max = logit_view.max_along_axis( 1 );
	tensor_t const shifted = logit_view.subtractColwise( row_max );
	m_probs                = shifted.cwiseExp();
	m_sum_exp   = m_probs.sum_along_axis( 1 );
	tensor_t const log_sum_exp = m_sum_exp.cwiseLog();
	tensor_t const log_p = shifted.subtractColwise( log_sum_exp );
	m_probs.divideRowsWithColInPlace( m_sum_exp );
	tensor_t const weighted = tgt_view * log_p;

	T const inv_N = T( 1 ) / static_cast<T>( N );
	T const loss  = -weighted.sum()*inv_N;

	m_grad = ( m_probs - tgt_view ) * inv_N;

	tensor_t grad_view( grad_out, N, C, false );
	grad_view = m_grad;

	return loss;
}

} // namespace refactor
} // namespace neural

#endif // REFACTOR_CROSS_ENTROPY_LOSS_HPP
