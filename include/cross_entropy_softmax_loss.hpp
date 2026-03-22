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
	m_inv_batch = static_cast<typename Tensor::value_type>( 1.0 ) / B;

	Tensor const row_max = out.max_along_axis( 1 );
	m_probs = out.subtractColwise( row_max );
	m_probs = m_probs.cwiseExp();
	m_sum_exp = m_probs.sum_along_axis( 1 );
	m_probs.divideRowsWithColInPlace( m_sum_exp );

	Tensor const weighted = target * m_probs.cwiseLog();
	return Tensor( 1u, 1u, weighted.asum() * m_inv_batch );
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
