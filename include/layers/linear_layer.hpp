#ifndef LINEAR_LAYER_HPP
#define LINEAR_LAYER_HPP
#include "layer_base.hpp"
#include "tensor_like.hpp"
#if NEURAL_CUDA_ENABLED
#include "cuda_tensor.hpp"
#include "neural_cuda_layer_sync.hpp"
#endif
#include <cstddef>
#include <random>
#include <stdexcept>

namespace neural {

template <typename Tensor>
class LinearLayer : public LayerBase<Tensor>
{
		static_assert( TensorLike<Tensor>, "Tensor must be a TensorLike type" );

	public:
		using tensor_t = Tensor;
		using bias_t = Tensor;

	public:
		/// Only output width is fixed; input width is taken from the input tensor on first forward.
		explicit LinearLayer( std::size_t out_features );
		LinearLayer( Tensor const &weights, Tensor const &bias );

		void initialize();

		std::size_t outFeatures() const;

		Tensor *forward();
		Tensor *backward();

		Tensor const &getGradWeights() const;
		Tensor const &getGradBias() const;
		Tensor &getWeights();
		Tensor &getBias();
		Tensor const &getWeights() const;
		Tensor const &getBias() const;

	private:
		void ensure_weights( std::size_t in_features );

		std::size_t m_out_features;
		Tensor m_weights;
		Tensor m_bias;

		Tensor m_grad_weights;
		Tensor m_grad_bias;
};

template < typename Tensor >
LinearLayer<Tensor>::LinearLayer( std::size_t out_features )
	: m_out_features( out_features )
{
}

template < typename Tensor >
LinearLayer<Tensor>::LinearLayer( Tensor const &weights, Tensor const &bias )
	: m_out_features( weights.cols() )
	, m_weights( weights )
	, m_bias( bias )
	, m_grad_weights( weights.rows(), weights.cols() )
	, m_grad_bias( bias.rows(), 1 )
{
}

template < typename Tensor >
std::size_t LinearLayer<Tensor>::outFeatures() const
{
	if ( m_weights.cols() > 0 ) {
		return m_weights.cols();
	}
	return m_out_features;
}

template < typename Tensor >
void LinearLayer<Tensor>::ensure_weights( std::size_t in_features )
{
	if ( m_weights.rows() > 0 ) {
		if ( m_weights.rows() != in_features ) {
			throw std::logic_error(
			    "LinearLayer: input width does not match allocated weight rows" );
		}
		return;
	}
	m_weights          = Tensor( in_features, m_out_features );
	m_grad_weights     = Tensor( in_features, m_out_features );
	m_bias             = Tensor( m_out_features, 1 );
	m_grad_bias        = Tensor( m_out_features, 1 );
	std::mt19937_64 wgen( std::random_device{}() );
	std::uint64_t const s = wgen();
	// He (uniform): U(-√(2/fan_in), √(2/fan_in)); bias left zero (constructors zero-init).
	m_weights.randomizeHe( in_features, s );
}

template < typename Tensor >
Tensor *LinearLayer<Tensor>::forward()
{
	ensure_weights( this->getInput()->cols() );
	this->getOutput()->matmulInPlace( *this->getInput(), m_weights );
	this->getOutput()->addColwiseInPlace( m_bias );

#if NEURAL_CUDA_ENABLED
	if constexpr ( is_cuda_tensor_v<Tensor> ) {
		cuda_layer_sync();
	}
#endif
	return this->getOutput();
}

template < typename Tensor >
Tensor *LinearLayer<Tensor>::backward()
{
	if ( m_weights.rows() == 0 ) {
		throw std::logic_error( "LinearLayer::backward: forward has not been run" );
	}
	Tensor const &grad = *this->getGradInput();
	m_grad_weights.matmulInPlace( *this->getInput(), grad, true );
	m_grad_bias.sumAlongAxisInPlace( grad, 0, true );
	this->getGradOutput()->matmulInPlace( grad, m_weights, false, true );

#if NEURAL_CUDA_ENABLED
	if constexpr ( is_cuda_tensor_v<Tensor> ) {
		cuda_layer_sync();
	}
#endif
	return this->getGradOutput();
}

template < typename Tensor >
Tensor const &LinearLayer<Tensor>::getGradWeights() const
{
	return m_grad_weights;
}

template < typename Tensor>
Tensor const &LinearLayer<Tensor>::getGradBias() const
{
	return m_grad_bias;
}

template < typename Tensor>
Tensor &LinearLayer<Tensor>::getWeights()
{
	return m_weights;
}

template < typename Tensor>
Tensor &LinearLayer<Tensor>::getBias()
{
	return m_bias;
}

template < typename Tensor>
Tensor const &LinearLayer<Tensor>::getWeights() const
{
	return m_weights;
}

template < typename Tensor>
Tensor const &LinearLayer<Tensor>::getBias() const
{
	return m_bias;
}

template < typename Tensor>
void LinearLayer<Tensor>::initialize()
{
	if ( m_weights.rows() == 0 ) {
		return;
	}
	std::mt19937_64 wgen( std::random_device{}() );
	std::uint64_t const s = wgen();
	m_weights.randomizeHe( m_weights.rows(), s );
}

} // namespace neural

#endif
