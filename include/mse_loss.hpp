#ifndef MSE_LOSS_HPP
#define MSE_LOSS_HPP
#include "tensor_like.hpp"
#include <stdexcept>

namespace neural {

template <typename Tensor>
class MSELoss
{
	static_assert( TensorLike<Tensor>, "Tensor must be a TensorLike type" );
	public:
		MSELoss();

		typename Tensor::value_type forward( Tensor const &pred, Tensor const &target );
		Tensor backward();

	private:
		Tensor m_diff;
		typename Tensor::value_type m_scale;
};

template <typename Tensor>
MSELoss<Tensor>::MSELoss()
{
}

template <typename Tensor>
typename Tensor::value_type MSELoss<Tensor>::forward( Tensor const &pred,
                                                      Tensor const &target )
{
	if ( pred.rows() != target.rows() || pred.cols() != target.cols() ) {
		throw std::invalid_argument(
		    "MSELoss: pred and target must have the same shape" );
	}

	m_diff = pred - target;
	auto const n = static_cast<typename Tensor::value_type>( pred.size() );
	m_scale = static_cast<typename Tensor::value_type>( 2.0 ) / n;

	// L = (1/n) * sum((pred - target)^2)
	auto squared = m_diff * m_diff;
	return squared.sum() / n;
}

template <typename Tensor>
Tensor MSELoss<Tensor>::backward()
{
	// dL/dpred = (2/n) * (pred - target)
	return m_diff * m_scale;
}

} // namespace neural

#endif
