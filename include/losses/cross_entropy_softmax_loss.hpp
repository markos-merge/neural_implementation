#ifndef CROSS_ENTROPY_SOFTMAX_LOSS_HPP
#define CROSS_ENTROPY_SOFTMAX_LOSS_HPP
#include "tensor_like.hpp"
#include <stdexcept>

namespace neural {

/// Fused softmax + categorical cross-entropy on **logits** (no separate SoftmaxLayer).
/// Forward uses numerically stable softmax; backward returns (p - target) / batch_size
/// for mean loss L = -(1/B) sum_b sum_i t_bi log p_bi.
///
/// `forward` returns a **1×1** tensor holding the loss. Reading a host scalar is explicit, e.g.
/// `loss.forward(...)(0,0)`.
template <typename Tensor>
class SoftmaxCrossEntropyLoss
{
	static_assert( TensorLike<Tensor>, "Tensor must be a TensorLike type" );
	public:
		using tensor_t = Tensor;
	public:
		SoftmaxCrossEntropyLoss() = default;

		Tensor forward( Tensor const &out, Tensor const &target );
		Tensor backward();

	private:
		Tensor m_probs;
		Tensor m_target;
		typename Tensor::value_type m_inv_batch;

		// cached tensors
		Tensor m_sum_exp;
};

template <typename Tensor>
Tensor SoftmaxCrossEntropyLoss<Tensor>::forward( Tensor const &out, Tensor const &target )
{
	if ( out.rows() != target.rows() || out.cols() != target.cols() ) {
		throw std::invalid_argument(
		    "SoftmaxCrossEntropyLoss: logits and target must have the same shape" );
	}
	if ( out.rows() == 0 ) {
		throw std::invalid_argument( "SoftmaxCrossEntropyLoss: empty batch" );
	}

	m_target = target;
	auto const B = static_cast<typename Tensor::value_type>( out.rows() );
	m_inv_batch = static_cast<typename Tensor::value_type>( 1.0 ) / static_cast<typename Tensor::value_type>( B );

	// Numerically stable log-softmax: log_p = (logits - row_max) - log(sum_exp).
	// Avoids `log(0)` when a non-argmax logit underflows after the shift, which
	// otherwise gives `0 * -inf = NaN` in the reduction even though probs are finite.
	Tensor const row_max = out.max_along_axis( 1 );
	Tensor const shifted = out.subtractColwise( row_max );
	m_probs = shifted.cwiseExp();
	m_sum_exp = m_probs.sum_along_axis( 1 );

	Tensor const log_sum_exp = m_sum_exp.cwiseLog();
	Tensor const log_p = shifted.subtractColwise( log_sum_exp );

	m_probs.divideRowsWithColInPlace( m_sum_exp );

	// L = -(1/B) * sum_{b,i} t_bi log p_bi = -(1/B) * algebraic sum of (target * log_p)
	Tensor const weighted = target * log_p;
	return Tensor( 1u, 1u, -weighted.sum() * m_inv_batch );
}

template <typename Tensor>
Tensor SoftmaxCrossEntropyLoss<Tensor>::backward()
{
	Tensor ret = m_probs;

	ret -= m_target;
	ret *= m_inv_batch;

	return ret;
}

} // namespace neural

#endif
