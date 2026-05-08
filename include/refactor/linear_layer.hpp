#ifndef REFACTOR_LINEAR_LAYER_HPP
#define REFACTOR_LINEAR_LAYER_HPP

#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "device.hpp"
#include "layer_base.hpp"
#include "cuda_tensor.hpp"

namespace neural {
namespace refactor {

template <typename T, typename Device = Cpu>
class LinearLayer : public LayerBase<T, Device>
{
public:
	explicit LinearLayer( std::size_t out_features );
	LinearLayer( T *weights, std::size_t in_features, std::size_t out_features, T *bias );

	std::size_t outFeatures() const;

	T *getWeights();
	T *getGradWeights();
	T *getBias();
	T *getGradBias();

	std::size_t numWeightParams() const noexcept { return m_weights.size(); }
	std::size_t numBiasParams() const noexcept { return m_bias.size(); }

	void forward();
	void backward();

private:
	using tensor_t = std::conditional_t<std::is_same_v<Device, Cuda>, CudaTensor<T>, Tensor<T>>;
	void ensureWeights( std::size_t in_features );

private:
	std::size_t m_out_features;
	tensor_t m_weights;
	tensor_t m_grad_weights;
	tensor_t m_bias;
	tensor_t m_grad_bias;
};

// Implementation
template <typename T, typename Device>
LinearLayer<T, Device>::LinearLayer( std::size_t out_features )
	: m_out_features( out_features )
{
}

template <typename T, typename Device>
LinearLayer<T, Device>::LinearLayer( T *weights, std::size_t in_features, std::size_t out_features, T *bias )
	: m_out_features( out_features )
	, m_weights( weights, in_features, out_features, true )
	, m_grad_weights( in_features, out_features )
	, m_bias( bias, out_features, 1, true )
	, m_grad_bias( out_features, 1 )
{
}

template <typename T, typename Device>
std::size_t LinearLayer<T, Device>::outFeatures() const
{
	if ( m_weights.rows() > 0 ) {
		return m_weights.cols();
	}
	return m_out_features;
}

template <typename T, typename Device>
void LinearLayer<T, Device>::ensureWeights( std::size_t in_features )
{
	if ( m_weights.rows() > 0 ) {
		if ( m_weights.rows() != in_features || m_weights.cols() != m_out_features ) {
			throw std::logic_error(
			    "LinearLayer: input width does not match allocated weight dimensions" );
		}
		return;
	}

	m_weights = tensor_t( in_features, m_out_features );
	m_grad_weights = tensor_t( in_features, m_out_features );
	m_bias = tensor_t( m_out_features, 1 );
	m_grad_bias = tensor_t( m_out_features, 1 );

	std::mt19937_64 wgen( std::random_device{}() );
	std::uint64_t const s = wgen();
	m_weights.randomizeHe( in_features, s );
}

template <typename T, typename Device>
T *LinearLayer<T, Device>::getWeights()
{
	return m_weights.data();
}

template <typename T, typename Device>
T *LinearLayer<T, Device>::getGradWeights()
{
	return m_grad_weights.data();
}

template <typename T, typename Device>
T *LinearLayer<T, Device>::getBias()
{
	return m_bias.data();
}

template <typename T, typename Device>
T *LinearLayer<T, Device>::getGradBias()
{
	return m_grad_bias.data();
}

template <typename T, typename Device>
void LinearLayer<T, Device>::forward()
{
	T *input = this->getInput()->fwdData();
	std::vector<std::size_t> const &input_shape = this->getInput()->shape();
	std::size_t const rows = input_shape[0];
	std::size_t const in_features = std::accumulate( input_shape.begin() + 1, input_shape.end(),
	    std::size_t{ 1 }, std::multiplies<std::size_t>() );

	ensureWeights( in_features );

	std::vector<std::size_t> const out_shape{ rows, m_out_features };
	if ( this->getOutput()->shape() != out_shape ) {
		this->getOutput()->resize( out_shape );
	}

	tensor_t in_view( input, rows, in_features, false );
	tensor_t out_view( this->getOutput()->fwdData(), rows, m_out_features, false );

	out_view.matmulInPlace( in_view, m_weights );
	out_view.addColwiseInPlace( m_bias );

#if NEURAL_CUDA_ENABLED
	if constexpr ( std::is_same_v<Device, Cuda> ) {
		cuda_layer_sync();
	}
#endif
}

template <typename T, typename Device>
void LinearLayer<T, Device>::backward()
{
	if ( m_weights.rows() == 0 ) {
		throw std::logic_error( "LinearLayer::backward: forward has not been run" );
	}

	T *input_act = this->getInput()->fwdData();
	std::vector<std::size_t> const &input_shape = this->getInput()->shape();
	std::size_t const rows = input_shape[0];
	std::size_t const in_features = std::accumulate( input_shape.begin() + 1, input_shape.end(),
	    std::size_t{ 1 }, std::multiplies<std::size_t>() );

	std::vector<std::size_t> const &upstream_shape = this->getGradInput()->shape();
	std::size_t const upstream_rows = upstream_shape[0];
	std::size_t const upstream_cols = std::accumulate(
	    upstream_shape.begin() + 1, upstream_shape.end(), std::size_t{ 1 },
	    std::multiplies<std::size_t>() );
	if ( upstream_rows != rows || upstream_cols != m_out_features ) {
		throw std::logic_error( "LinearLayer::backward: gradient shape does not match forward output" );
	}

	if ( this->getGradOutput()->shape() != input_shape ) {
		this->getGradOutput()->resize( input_shape );
	}

	tensor_t const in_view( input_act, rows, in_features, false );
	tensor_t const grad_upstream( this->getGradInput()->fwdData(), rows, m_out_features, false );

	m_grad_weights.matmulInPlace( in_view, grad_upstream, true, false );

	m_grad_bias.sumAlongAxisInPlace( grad_upstream, 0, true );

	T *grad_prior = this->getGradOutput()->fwdData();
	tensor_t grad_prior_view( grad_prior, rows, in_features, false );
	grad_prior_view.matmulInPlace( grad_upstream, m_weights, false, true );

#if NEURAL_CUDA_ENABLED
	if constexpr ( std::is_same_v<Device, Cuda> ) {
		cuda_layer_sync();
	}
#endif
}

} // namespace refactor
} // namespace neural

#endif // REFACTOR_LINEAR_LAYER_HPP
