#ifndef DROPOUT_LAYER_HPP
#define DROPOUT_LAYER_HPP

#include "layer_base.hpp"
#if NEURAL_CUDA_ENABLED
#include "cuda_tensor.hpp"
#include "neural_cuda_layer_sync.hpp"
#endif
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace neural {

template <typename Tensor>
class DropoutLayer : public LayerBase<Tensor>
{
	public:
		using tensor_t = Tensor;
		using value_type = typename Tensor::value_type;

	public:
		/// \p keep_prob must be in (0, 1]. \p rng_seed seeds the internal \c std::mt19937.
		/// Use \ref setTraining(false) for inference (no mask, no scaling).
		explicit DropoutLayer( value_type keep_prob, std::uint32_t rng_seed );

		value_type keepProb() const;

		void setTraining( bool training ) noexcept;
		bool training() const noexcept;

		/// Reseed the internal RNG (e.g. for repeatable tests).
		void setRngSeed( std::uint32_t seed );

		Tensor *forward();
		Tensor *backward();

	private:
		void findNewMaskNApply() const;
		void sample_mask_and_forward_train();

		value_type m_keep_prob;
		value_type m_inv_keep_prob;
		bool m_training = true;
		std::mt19937 m_rng;
		Tensor m_mask_cache;
		Tensor m_mask;
		std::uint32_t m_seed;
};

template <typename Tensor>
DropoutLayer<Tensor>::DropoutLayer( value_type keep_prob, std::uint32_t rng_seed )
	: m_keep_prob( keep_prob )
	, m_inv_keep_prob( static_cast<value_type>( 1 ) / keep_prob )
	, m_rng( rng_seed )
	, m_seed( rng_seed )
{
	findNewMaskNApply();
}

template <typename Tensor>
typename Tensor::value_type DropoutLayer<Tensor>::keepProb() const
{
	return m_keep_prob;
}

template <typename Tensor>
void DropoutLayer<Tensor>::setTraining( bool training ) noexcept
{
	m_training = training;
}

template <typename Tensor>
bool DropoutLayer<Tensor>::training() const noexcept
{
	return m_training;
}

template <typename Tensor>
void DropoutLayer<Tensor>::setRngSeed( std::uint32_t seed )
{
	m_rng.seed( seed );
	m_seed = seed;
}

template <typename Tensor>
void DropoutLayer<Tensor>::findNewMaskNApply() const
{
	if ( !( m_keep_prob > static_cast<value_type>( 0 ) )
	     || m_keep_prob > static_cast<value_type>( 1 ) ) {
		throw std::invalid_argument(
		    "DropoutLayer: keep_prob must be in (0, 1]" );
	}
	if ( !std::isfinite( static_cast<double>( m_keep_prob ) ) ) {
		throw std::invalid_argument( "DropoutLayer: keep_prob must be finite" );
	}
}

template <typename Tensor>
void DropoutLayer<Tensor>::sample_mask_and_forward_train()
{
	Tensor *input = this->getInput();
	Tensor *out = this->getOutput();
	if ( m_mask.shape() != input->shape() ) {
		m_mask = Tensor( input->shape() );
		m_mask_cache = Tensor( input->shape() );
	}

	std::uniform_int_distribution<std::uint64_t> seed_dist(
	    0, std::numeric_limits<std::uint64_t>::max() );
	std::uint64_t const seed = seed_dist( m_rng );
	m_mask_cache.randomize( seed );

	value_type const thresh = static_cast<value_type>( 1 ) - m_keep_prob;
	m_mask.cwiseGreaterInPlace( m_mask_cache, thresh );
	m_mask *= m_inv_keep_prob;
	input->elementwiseMultiply( m_mask, *out );
}

template <typename Tensor >
Tensor *DropoutLayer<Tensor>::forward()
{
	// std::cout << "Seed: " << m_seed << " DropoutLayer::forward: m_training=" << m_training << " m_keep_prob=" << m_keep_prob << "\n";
	// std::this_thread::sleep_for( std::chrono::seconds( 1 ) );
	Tensor *input = this->getInput();
	Tensor *output = this->getOutput();

	if ( !m_training || m_keep_prob == static_cast<value_type>( 1 ) ) {
		if ( output->shape() != input->shape() ) {
			*output = Tensor( input->shape() );
		}
		*output = *input;
	} else {
		if ( output->shape() != input->shape() ) {
			*output = Tensor( input->shape() );
		}
		sample_mask_and_forward_train();
	}

#if NEURAL_CUDA_ENABLED
	if constexpr ( is_cuda_tensor_v<Tensor> ) {
		cuda_layer_sync();
	}
#endif
	return output;
}

template <typename Tensor >
Tensor *DropoutLayer<Tensor>::backward()
{
	Tensor *d_in = this->getGradInput();
	Tensor *d_out = this->getGradOutput();

	if ( d_out->shape() != d_in->shape() ) {
		*d_out = Tensor( d_in->shape() );
	}

	if ( !m_training ) {
		*d_out = *d_in;
	} else if ( m_keep_prob == static_cast<value_type>( 1 ) ) {
		*d_out = *d_in;
	} else {
		*d_out = *d_in;
		*d_out *= m_mask;
	}

#if NEURAL_CUDA_ENABLED
	if constexpr ( is_cuda_tensor_v<Tensor> ) {
		cuda_layer_sync();
	}
#endif
	return d_out;
}

} // namespace neural

#endif
