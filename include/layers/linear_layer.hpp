#ifndef LINEAR_LAYER_HPP
#define LINEAR_LAYER_HPP
#include "layer_base.hpp"
#include "tensor_like.hpp"

namespace neural {

template <typename Tensor>
class LinearLayer : public LayerBase< Tensor >
{
	static_assert( TensorLike<Tensor>, "Tensor must be a TensorLike type" );
	public:
		LinearLayer( std::size_t in_features, std::size_t out_features );
		LinearLayer( Tensor const &weights, Tensor const &bias );

		void initialize();

		Tensor *forward();
		Tensor *backward();

		Tensor const &getGradWeights() const;
		Tensor const &getGradBias() const;
		Tensor &getWeights();
		Tensor &getBias();
		Tensor const &getWeights() const;
		Tensor const &getBias() const;
	private:
		Tensor m_weights;
		Tensor m_bias;

		Tensor m_grad_weights;
		Tensor m_grad_bias;
};

template <typename Tensor >
LinearLayer<Tensor>::LinearLayer( std::size_t in_features, std::size_t out_features )
	: m_weights( in_features, out_features )
	, m_bias( Tensor( out_features, 1, static_cast<typename Tensor::value_type>(0.) ) )
	, m_grad_weights( in_features, out_features )
	, m_grad_bias( out_features, 1 )
{
	initialize();
}

template <typename Tensor >
LinearLayer<Tensor>::LinearLayer( Tensor const &weights, Tensor const &bias )
	: m_weights( weights )
	, m_bias( bias )
	, m_grad_weights( weights.rows(), weights.cols() )
	, m_grad_bias( bias.rows(), 1 )
{
}

template <typename Tensor >
Tensor *LinearLayer<Tensor>::forward()
{
	this->getOutput()->matmulInPlace( *this->getInput(), m_weights );
	this->getOutput()->addColwiseInPlace( m_bias );

	return this->getOutput();
}

template <typename Tensor >
Tensor *LinearLayer<Tensor>::backward()
{
	Tensor const &grad = *this->getGradInput();
	m_grad_weights.matmulInPlace( *this->getInput(), grad, true );
	m_grad_bias.sumAlongAxisInPlace( grad, 0, true );
	this->getGradOutput()->matmulInPlace( grad, m_weights, false, true );

	return this->getGradOutput();
}

template <typename Tensor >
Tensor const &LinearLayer<Tensor>::getGradWeights() const
{
	return m_grad_weights;
}

template <typename Tensor>
Tensor const &LinearLayer<Tensor>::getGradBias() const
{
	return m_grad_bias;
}

template <typename Tensor>
Tensor &LinearLayer<Tensor>::getWeights()
{
	return m_weights;
}

template <typename Tensor>
Tensor &LinearLayer<Tensor>::getBias()
{
	return m_bias;
}

template <typename Tensor>
Tensor const &LinearLayer<Tensor>::getWeights() const
{
	return m_weights;
}

template <typename Tensor>
Tensor const &LinearLayer<Tensor>::getBias() const
{
	return m_bias;
}

template <typename Tensor>
void LinearLayer<Tensor>::initialize()
{
	m_weights.randomizeHe( m_weights.rows() );
	m_bias = Tensor( m_weights.cols(), 1, static_cast<typename Tensor::value_type>(0.) );
}

} // namespace neural

#endif
