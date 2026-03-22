#ifndef LINEAR_LAYER_HPP
#define LINEAR_LAYER_HPP
#include "tensor_like.hpp"

namespace neural {

template <typename Tensor>
class LinearLayer
{
		static_assert( TensorLike<Tensor>, "Tensor must be a TensorLike type" );
	public:
		LinearLayer( std::size_t in_features, std::size_t out_features );
		LinearLayer( Tensor const &weights, Tensor const &bias );

		void initialize();

		Tensor forward( Tensor const &input );
		Tensor backward( Tensor const &grad_output );

		Tensor const &getGradWeights() const;
		Tensor const &getGradBias() const;
		/// I did return the weights and bias by reference for performance reasons
		Tensor &getWeights();
		Tensor &getBias();
		Tensor const &getWeights() const;
		Tensor const &getBias() const;
	private:
		Tensor m_weights; // (in_features, out_features)
		Tensor m_bias;    // ( out_features, 1)

	// Cache for backward pass
		Tensor m_input;
		Tensor m_grad_weights;
		Tensor m_grad_bias;
};

template <typename Tensor >
LinearLayer<Tensor>::LinearLayer( std::size_t in_features, std::size_t out_features )
	: m_weights( in_features, out_features )
	, m_bias( Tensor( out_features, 1, static_cast<typename Tensor::value_type>(0.) ) )
{
	initialize();
}

template <typename Tensor >
LinearLayer<Tensor>::LinearLayer( Tensor const &weights, Tensor const &bias )
	: m_weights( weights )
	, m_bias( bias )
{
}

template <typename Tensor >
Tensor LinearLayer<Tensor>::forward( Tensor const &input )
{
	Tensor ret = input.matmul( m_weights );

	m_input = input;

	ret.addColwiseInPlace( m_bias );

	return ret;
}

template <typename Tensor >
Tensor LinearLayer<Tensor>::backward( Tensor const &grad_output )
{
	m_grad_weights = m_input.transpose().matmul( grad_output );
	m_grad_bias = grad_output.sum_along_axis( 0 ).transpose();

	return grad_output.matmul( m_weights.transpose() );
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