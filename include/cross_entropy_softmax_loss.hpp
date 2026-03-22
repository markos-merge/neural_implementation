#ifndef CROSS_ENTROPY_SOFTMAX_LOSS_HPP
#define CROSS_ENTROPY_SOFTMAX_LOSS_HPP
#include "tensor_like.hpp"
#include <stdexcept>

namespace neural {

/// Fused softmax + categorical cross-entropy on **logits** (no separate SoftmaxLayer).
/// Forward uses numerically stable log-softmax; backward returns (p - target) / batch_size
/// for mean loss L = -(1/B) sum_b sum_i t_bi log p_bi.
template <typename Tensor>
class SoftmaxCrossEntropyLoss
{
	static_assert( TensorLike<Tensor>, "Tensor must be a TensorLike type" );
	public:
		SoftmaxCrossEntropyLoss();

		typename Tensor::value_type forward( Tensor const &logits, Tensor const &target );
		Tensor backward();

	private:
		Tensor m_probs;
		Tensor m_target;
		typename Tensor::value_type m_inv_batch;
};

template <typename Tensor>
SoftmaxCrossEntropyLoss<Tensor>::SoftmaxCrossEntropyLoss()
{
}

template <typename Tensor>
typename Tensor::value_type
SoftmaxCrossEntropyLoss<Tensor>::forward( Tensor const &logits, Tensor const &target )
{
	if ( logits.rows() != target.rows() || logits.cols() != target.cols() ) {
		throw std::invalid_argument(
		    "SoftmaxCrossEntropyLoss: logits and target must have the same shape" );
	}
	if ( logits.rows() == 0 ) {
		throw std::invalid_argument( "SoftmaxCrossEntropyLoss: empty batch" );
	}

	m_target = target;
	auto const B = static_cast<typename Tensor::value_type>( logits.rows() );
	m_inv_batch = static_cast<typename Tensor::value_type>( 1.0 ) / B;

	Tensor const row_max = logits.max_along_axis( 1 );
	Tensor shifted = logits.subtractColwise( row_max );
	Tensor const exp_shifted = shifted.cwiseExp();
	Tensor const sum_exp = exp_shifted.sum_along_axis( 1 );
	m_probs = exp_shifted;
	m_probs.divideRowsWithColInPlace( sum_exp );

	// log p_i = (x_i - m) - log(sum_k exp(x_k - m)) = shifted_i - log(S)
	Tensor const log_sum_exp = sum_exp.cwiseLog();
	Tensor const log_probs = shifted.subtractColwise( log_sum_exp );

	Tensor const weighted = target * log_probs;
	return -weighted.sum() * m_inv_batch;
}

template <typename Tensor>
Tensor SoftmaxCrossEntropyLoss<Tensor>::backward()
{
	return ( m_probs - m_target ) * m_inv_batch;
}

} // namespace neural

#endif
