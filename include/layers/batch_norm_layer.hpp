#ifndef BATCH_NORM_LAYER_HPP
#define BATCH_NORM_LAYER_HPP

#include <cstddef>
#include "layer_base.hpp"
#include <array>
#if NEURAL_CUDA_ENABLED
#include "cuda_tensor_n.hpp"
#include "neural_cuda_layer_sync.hpp"
#endif

namespace neural {

/// Spatial batch normalization for NCHW tensors (normalize per channel over \f$N \times H \times W\f$).
/// Forward delegates normalization math to tensor operations.
///
/// Typical placement: **Conv → BatchNorm → ReLU** (normalize conv output before nonlinearity).
///
/// Training: batch mean / variance; backward couples samples in the batch.  
/// Evaluation: fixed running mean / variance (updated during training with EMA).
template <typename TensorN_t>
class BatchNormLayer : public LayerBase<TensorN_t>
{
	static_assert( TensorN_t::rank_v == 4, "BatchNormLayer expects rank-4 (NCHW) tensors" );

	public:
		using tensor_t = TensorN_t;
		using value_type = typename TensorN_t::value_type;

		/// \p num_features is \f$C\f$ (channel count). \p eps avoids divide-by-zero.
		/// \p momentum matches PyTorch / cuDNN: \(\text{running} \leftarrow (1-m)\,\text{running} + m\,\text{batch}\).
		BatchNormLayer( std::size_t num_features, value_type eps, value_type momentum );

		std::size_t numFeatures() const { return m_num_features; }
		bool nonRegularizable() const noexcept { return true; }

		void setTraining( bool training ) { m_training = training; }
		bool isTraining() const { return m_training; }

		/// Learnable scale \f$\gamma\f$ (per channel), shape \f$\{1, C, 1, 1\}\f$.
		TensorN_t       &getGamma() { return m_gamma; }
		TensorN_t const &getGamma() const { return m_gamma; }
		/// Learnable shift \f$\beta\f$ (per channel), shape \f$\{1, C, 1, 1\}\f$.
		TensorN_t       &getBeta() { return m_beta; }
		TensorN_t const &getBeta() const { return m_beta; }
		TensorN_t       &getGradGamma() { return m_grad_gamma; }
		TensorN_t const &getGradGamma() const { return m_grad_gamma; }
		TensorN_t       &getGradBeta() { return m_grad_beta; }
		TensorN_t const &getGradBeta() const { return m_grad_beta; }

		TensorN_t       &getWeights()      { return m_gamma; }
		TensorN_t const &getWeights() const { return m_gamma; }
		TensorN_t       &getGradWeights()      { return m_grad_gamma; }
		TensorN_t const &getGradWeights() const { return m_grad_gamma; }
		TensorN_t       &getBias()      { return m_beta; }
		TensorN_t const &getBias() const { return m_beta; }
		TensorN_t       &getGradBias()      { return m_grad_beta; }
		TensorN_t const &getGradBias() const { return m_grad_beta; }

		TensorN_t *forward();
		TensorN_t *backward();

	private:
		std::size_t m_num_features;
		value_type m_eps;
		value_type m_momentum;

		bool m_training = true;

		TensorN_t m_gamma;//per-channel scale
		TensorN_t m_beta;//per-channel shift
		TensorN_t m_grad_gamma;
		TensorN_t m_grad_beta;

		/// Running mean / variance (EMA), per channel; used when \ref m_training is false.
		TensorN_t m_last_running_mean;
		TensorN_t m_last_running_var;
		TensorN_t m_running_mean;
		TensorN_t m_running_var;

		// Saved from forward for backward (implementation will fill these):
		// e.g. batch mean, batch var, normalized x_hat, or workspace buffers.
};

template <typename TensorN_t>
BatchNormLayer<TensorN_t>::BatchNormLayer( std::size_t num_features, value_type eps, value_type momentum )
	: m_num_features( num_features )
	, m_eps( eps )
	, m_momentum( momentum )
{
	// Implement: allocate \c m_gamma / \c m_beta with shape \f$\{1,C,1,1\}\f$, init \f$\gamma=1,\beta=0\f$;
	// allocate \c m_running_mean / \c m_running_var; scratch for forward/backward.
}

template <typename TensorN_t>
TensorN_t *BatchNormLayer<TensorN_t>::forward()
{
	using value_type = typename TensorN_t::value_type;
	bool is_first_forward = false;

	//calc mean
	if( m_running_mean.size() != this->getInput()->shape()[1] ) {
		std::array< std::size_t, 4 > const mean_shape = { 1, this->getInput()->shape()[1], 1, 1 };
		m_running_mean = TensorN_t( mean_shape );
		m_last_running_mean = TensorN_t( mean_shape );
	}
	
	if( m_running_var.size() != this->getInput()->shape()[1] ) {
		std::array< std::size_t, 4 > const var_shape = { 1, this->getInput()->shape()[1], 1, 1 };
		m_running_var = TensorN_t( var_shape );
		m_last_running_var = TensorN_t( var_shape, static_cast<value_type>( 1 ) );
	}
	
	if( m_gamma.size() != this->getInput()->shape()[1] ) {
		std::array< std::size_t, 4 > const gamma_shape = { 1, this->getInput()->shape()[1], 1, 1 };
		m_gamma = TensorN_t( gamma_shape, static_cast<value_type>( 1 ) );
		is_first_forward = true;
	}

	if( m_beta.size() != this->getInput()->shape()[1] ) {
		std::array< std::size_t, 4 > const beta_shape = { 1, this->getInput()->shape()[1], 1, 1 };
		m_beta = TensorN_t( beta_shape, static_cast<value_type>( 0 ) );
		is_first_forward = true;
	}
	
	if( this->getOutput()->shape() != this->getInput()->shape() ) {
		this->getOutput()->throw_if_non_owning_realloc(
		    "BatchNormLayer::forward: cannot reallocate non-owning output" );
		TensorN_t const fresh( this->getInput()->shape() );
		*this->getOutput() = fresh;
		is_first_forward = true;
	}

	batch_norm_forward_nchw( *this->getInput(), m_gamma, m_beta, m_running_mean,
	                         m_running_var, m_last_running_mean, m_last_running_var,
	                         m_eps, m_momentum, m_training, is_first_forward,
	                         *this->getOutput() );

#if NEURAL_CUDA_ENABLED
	if constexpr ( is_cuda_tensor4_v<TensorN_t> ) {
		cuda_layer_sync();
	}
#endif
	return this->getOutput();
}

template <typename TensorN_t>
TensorN_t *BatchNormLayer<TensorN_t>::backward()
{
	if( this->getGradOutput()->shape() != this->getGradInput()->shape() ) {
		this->getGradOutput()->throw_if_non_owning_realloc(
		    "BatchNormLayer::backward: cannot reallocate non-owning grad output" );
		TensorN_t const fresh( this->getGradInput()->shape() );
		*this->getGradOutput() = fresh;
	}

	if( this->m_grad_beta.size() != this->getGradInput()->shape()[1] ) {
		std::array< std::size_t, 4 > const beta_shape = { 1, this->getGradInput()->shape()[1], 1, 1 };
		m_grad_beta = TensorN_t( beta_shape );
	}

	if( this->m_grad_gamma.size() != this->getGradInput()->shape()[1] ) {
		std::array< std::size_t, 4 > const gamma_shape = { 1, this->getGradInput()->shape()[1], 1, 1 };
		m_grad_gamma = TensorN_t( gamma_shape );
	}

	batch_norm_backward_nchw( *this->getGradInput(), *this->getInput(), m_gamma,
	                          m_running_mean, m_running_var, m_eps, m_grad_gamma,
	                          m_grad_beta, *this->getGradOutput() );

#if NEURAL_CUDA_ENABLED
	if constexpr ( is_cuda_tensor4_v<TensorN_t> ) {
		cuda_layer_sync();
	}
#endif
	return this->getGradOutput();
}

} // namespace neural

#endif
