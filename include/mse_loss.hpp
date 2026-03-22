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

		/// Returns a 1×1 tensor with mean squared error.
		Tensor forward( Tensor const &pred, Tensor const &target );
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
Tensor MSELoss<Tensor>::forward( Tensor const &pred, Tensor const &target )
{
	if ( pred.rows() != target.rows() || pred.cols() != target.cols() ) {
		throw std::invalid_argument(
		    "MSELoss: pred and target must have the same shape" );
	}

	m_diff = pred - target;
	auto const n = static_cast<typename Tensor::value_type>( pred.size() );
	m_scale = static_cast<typename Tensor::value_type>( 2.0 ) / n;

	// L = (1/n) * sum((pred - target)^2)
	Tensor const squared = m_diff * m_diff;
	auto const inv_n = static_cast<typename Tensor::value_type>( 1.0 ) / n;
	return Tensor( 1u, 1u, squared.sum() * inv_n );
}

template <typename Tensor>
Tensor MSELoss<Tensor>::backward()
{
	// dL/dpred = (2/n) * (pred - target)
	return m_diff * m_scale;
}

} // namespace neural

#endif
