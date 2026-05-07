#ifndef REFACTOR_DROPOUT_LAYER_HPP
#define REFACTOR_DROPOUT_LAYER_HPP

#include <cstdint>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include "layer_base.hpp"
#include "latch_layer.hpp"
#include "tensor.hpp"
#include <type_traits>
#if NEURAL_CUDA_ENABLED
#include "cuda_tensor.hpp"
#include "neural_cuda_layer_sync.hpp"
#endif

namespace neural {
namespace refactor {

template <typename T, typename Device = Cpu>
class DropoutLayer : public LayerBase<T, Device>
{
public:
	explicit DropoutLayer( T keep_prob, std::uint32_t rng_seed = std::random_device{}() );

	T    keepProb()                 const;
	void setTraining( bool training ) noexcept;
	bool training()                 const noexcept;
	void setRngSeed( std::uint32_t seed );

	void forward();
	void backward();

private:
	void validateKeepProb() const;

	using tensor_t = std::conditional_t<std::is_same_v< Device, Cuda >, CudaTensor<T>, Tensor<T>>;
	T             m_keep_prob;
	T             m_inv_keep_prob;
	bool          m_training = true;
	std::uint32_t m_seed{};
	std::mt19937  m_rng;
	tensor_t      m_mask;
};

template <typename T, typename Device>
void DropoutLayer<T, Device>::validateKeepProb() const
{
	if ( !( m_keep_prob > static_cast<T>( 0 ) ) || m_keep_prob > static_cast<T>( 1 ) ) {
		throw std::invalid_argument( "DropoutLayer: keep_prob must be in (0, 1]" );
	}
	if ( !std::isfinite( static_cast<double>( m_keep_prob ) ) ) {
		throw std::invalid_argument( "DropoutLayer: keep_prob must be finite" );
	}
}

template <typename T, typename Device>
DropoutLayer<T, Device>::DropoutLayer( T keep_prob, std::uint32_t rng_seed )
	: m_keep_prob( keep_prob )
	, m_inv_keep_prob( static_cast<T>( 1 ) / keep_prob )
	, m_seed( rng_seed )
	, m_rng( rng_seed )
{
	validateKeepProb();
}

template <typename T, typename Device>
T DropoutLayer<T, Device>::keepProb() const
{
	return m_keep_prob;
}

template <typename T, typename Device>
void DropoutLayer<T, Device>::setTraining( bool training ) noexcept
{
	m_training = training;
}

template <typename T, typename Device>
bool DropoutLayer<T, Device>::training() const noexcept
{
	return m_training;
}

template <typename T, typename Device>
void DropoutLayer<T, Device>::setRngSeed( std::uint32_t seed )
{
	m_rng.seed( seed );
	m_seed = seed;
}

//Implementation
template <typename T, typename Device>
void DropoutLayer<T, Device>::forward()
{
	T *input = this->getInput()->fwdData();
	std::vector<std::size_t> const &input_shape = this->getInput()->shape();
	std::size_t const rows = input_shape[0];
	std::size_t const cols =
	    std::accumulate( input_shape.begin() + 1, input_shape.end(),
	                     std::size_t{ 1 }, std::multiplies<std::size_t>() );

	if ( this->getOutput()->shape() != input_shape ) {
		this->getOutput()->resize( input_shape );
	}

	if( !m_training || m_keep_prob == static_cast<T>(1) ) {
		tensor_t in_view( input, rows, cols, false );
		tensor_t out_view( this->getOutput()->fwdData(), rows, cols, false );
		out_view = in_view;
	} else {
		std::uniform_int_distribution<std::uint64_t> seed_dist(
		    0, std::numeric_limits<std::uint64_t>::max() );
		std::uint64_t const seed = seed_dist( m_rng );
		T const thresh = static_cast<T>( 1 ) - m_keep_prob;

		tensor_t rnd( rows, cols );
		rnd.randomize( seed );

		if ( m_mask.rows() != rows || m_mask.cols() != cols ) {
			m_mask = tensor_t( rows, cols );
		}
		m_mask.cwiseGreaterInPlace( rnd, thresh );
		m_mask *= m_inv_keep_prob;

		tensor_t in_view( input, rows, cols, false );
		tensor_t out_view( this->getOutput()->fwdData(), rows, cols, false );

		in_view.elementwiseMultiply( m_mask, out_view );
	}

#if NEURAL_CUDA_ENABLED
	if constexpr ( std::is_same_v<Device, Cuda> ) {
		cuda_layer_sync();
	}
#endif
}

template <typename T, typename Device>
void DropoutLayer<T, Device>::backward()
{
	T *grad_input = this->getGradInput()->fwdData();
	std::vector<std::size_t> const &grad_input_shape = this->getGradInput()->shape();
	std::size_t const rows = grad_input_shape[0];
	std::size_t const cols = std::accumulate( grad_input_shape.begin() + 1, grad_input_shape.end(),
		std::size_t{1}, std::multiplies<std::size_t>() );

	if( this->getGradOutput()->shape() != grad_input_shape ) {
		this->getGradOutput()->resize( grad_input_shape );
	}

	if ( !m_training || m_keep_prob == static_cast<T>( 1 ) ) {
		tensor_t grad_input_view( grad_input, rows, cols, false );
		tensor_t grad_output_view( this->getGradOutput()->fwdData(), rows, cols, false );
		grad_output_view = grad_input_view;
	} else {
		tensor_t grad_input_view( grad_input, rows, cols, false );
		tensor_t grad_output_view( this->getGradOutput()->fwdData(), rows, cols, false );
		grad_input_view.elementwiseMultiply( m_mask, grad_output_view );
	}

#if NEURAL_CUDA_ENABLED
	if constexpr ( std::is_same_v<Device, Cuda> ) {
		cuda_layer_sync();
	}
#endif
}
} // namespace refactor
} // namespace neural

#endif // REFACTOR_DROPOUT_LAYER_HPP
