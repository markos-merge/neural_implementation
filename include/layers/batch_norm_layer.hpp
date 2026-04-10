#ifndef BATCH_NORM_LAYER_HPP
#define BATCH_NORM_LAYER_HPP

#include <cstddef>
#include "layer_base.hpp"
#include <numeric>
#include <algorithm>
#include <cmath>
#include <array>

namespace neural {

/// Spatial batch normalization for NCHW tensors (normalize per channel over \f$N \times H \times W\f$).
/// **Forward / backward are not implemented** — this file defines the type and wiring only.
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

		/// \p num_features is \f$C\f$ (channel count). \p eps avoids divide-by-zero; \p momentum is the EMA factor for running stats.
		BatchNormLayer( std::size_t num_features, value_type eps, value_type momentum );

		std::size_t numFeatures() const { return m_num_features; }

		void set_training( bool training ) { m_training = training; }
		bool is_training() const { return m_training; }

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
		TensorN_t m_aux_tensor;

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
		m_last_running_var = TensorN_t( var_shape );
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
		*this->getOutput() = TensorN_t( this->getInput()->shape() );
		is_first_forward = true;
	}
	
	this->getInput()->swapAxes( { 1, 0, 2, 3 }, m_aux_tensor );
	auto strides = m_aux_tensor.strides();
	auto aux_shape = m_aux_tensor.shape();
	value_type *aux_data = m_aux_tensor.data();
	value_type *running_mean_data = m_running_mean.data();
	value_type *running_var_data = m_running_var.data();

	if ( m_training || is_first_forward ) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for ( std::size_t c = 0; c < aux_shape[0]; ++c ) {
			value_type channel_sum = std::accumulate(
			    aux_data + c * strides[0],
			    aux_data + c * strides[0] + aux_shape[1] * strides[1],
			    static_cast<value_type>( 0.0 ) );
			channel_sum /=
			    static_cast<value_type>( aux_shape[1] * aux_shape[2] * aux_shape[3] );
			running_mean_data[c] = channel_sum;
		}

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for ( std::size_t c = 0; c < aux_shape[0]; ++c ) {
			value_type channel_sum = std::accumulate(
			    aux_data + c * strides[0],
			    aux_data + c * strides[0] + aux_shape[1] * strides[1],
			    static_cast<value_type>( 0.0 ),
			    [&]( value_type acc, value_type val ) {
				    return acc + ( val - running_mean_data[c] ) *
				                     ( val - running_mean_data[c] );
			    } );
			channel_sum /=
			    static_cast<value_type>( aux_shape[1] * aux_shape[2] * aux_shape[3] );
			running_var_data[c] = channel_sum;
		}
	} else {
		m_running_mean = m_last_running_mean;
		m_running_var = m_last_running_var;
	}


	auto input_shape = this->getInput()->shape();
	auto input_strides = this->getInput()->strides();
	value_type *input_data = this->getInput()->data();
	value_type *output_data = this->getOutput()->data();
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
	for ( std::size_t b = 0; b < input_shape[0]; ++b ) {
		for ( std::size_t c = 0; c < input_shape[1]; ++c ) {
			double mean_channel = running_mean_data[c];
			double var_channel = std::sqrt( running_var_data[c] + this->m_eps );
			double gamma = m_gamma.at( 0, c, 0, 0 );
			double beta = m_beta.at( 0, c, 0, 0 );
			std::transform( input_data + b * input_strides[0] + c * input_strides[1],
			                input_data + b * input_strides[0] + c * input_strides[1] +
			                    input_shape[2] * input_shape[3],
			                output_data + b * input_strides[0] + c * input_strides[1],
			                [&]( value_type val ) {
				                return gamma * ( val - mean_channel ) / var_channel +
				                       beta;
			                } );
		}
	}

	if( is_first_forward ) {
		m_last_running_mean = m_running_mean;
		m_last_running_var = m_running_var;
	} else if( m_training ) {
		for( std::size_t c = 0; c < m_last_running_mean.size(); ++c ) {
			m_last_running_mean.data()[c] = m_last_running_mean.data()[c] * m_momentum + running_mean_data[c] * ( 1.0 - m_momentum );
			m_last_running_var.data()[c] = m_last_running_var.data()[c] * m_momentum + running_var_data[c] * ( 1.0 - m_momentum );
		}
	}

	return this->getOutput();
}

template <typename TensorN_t>
TensorN_t *BatchNormLayer<TensorN_t>::backward()
{
	if( this->getGradOutput()->shape() != this->getGradInput()->shape() ) {
		*this->getGradOutput() = TensorN_t( this->getGradInput()->shape() );
	}

	if( this->m_grad_beta.size() != this->getGradInput()->shape()[1] ) {
		std::array< std::size_t, 4 > const beta_shape = { 1, this->getGradInput()->shape()[1], 1, 1 };
		m_grad_beta = TensorN_t( beta_shape );
	}

	if( this->m_grad_gamma.size() != this->getGradInput()->shape()[1] ) {
		std::array< std::size_t, 4 > const gamma_shape = { 1, this->getGradInput()->shape()[1], 1, 1 };
		m_grad_gamma = TensorN_t( gamma_shape );
	}
	
	auto in_shape = this->getGradInput()->shape();
	auto in_strides = this->getGradInput()->strides();

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for ( std::size_t c = 0; c < in_shape[1]; ++c ) {
		value_type beta_grad = static_cast<value_type>( 0 );
		for ( std::size_t b = 0; b < in_shape[0]; ++b ) {
			beta_grad += std::accumulate(
			    this->getGradInput()->data() + b * in_strides[0] + c * in_strides[1],
			    this->getGradInput()->data() + b * in_strides[0] + c * in_strides[1] +
			        in_shape[2] * in_shape[3],
			    static_cast<value_type>( 0.0 ) );
		}
		m_grad_beta.data()[c] = beta_grad;
	}

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for ( std::size_t c = 0; c < in_shape[1]; ++c ) {
		value_type gamma_grad = static_cast<value_type>( 0 );
		for ( std::size_t b = 0; b < in_shape[0]; ++b ) {
			for ( std::size_t i = 0; i < in_strides[1]; ++i ) {
				std::size_t const idx = b * in_strides[0] + c * in_strides[1] + i;
				gamma_grad += this->getGradInput()->data()[idx] * ( this->getInput()->data()[idx] - m_running_mean.data()[c] ) / std::sqrt( m_running_var.data()[c] + this->m_eps );
			}
		}
		m_grad_gamma.data()[c] = gamma_grad;
	}

	value_type *grad_output_data = this->getGradOutput()->data();
	value_type const *input_data = this->getInput()->data();
	value_type const *dy_data = this->getGradInput()->data();
	auto out_strides = this->getGradOutput()->strides();
	std::size_t const M = in_shape[0] * in_strides[1];

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for ( std::size_t c = 0; c < in_shape[1]; ++c ) {
		value_type const inv_std = static_cast<value_type>( 1 ) / std::sqrt( m_running_var.data()[c] + this->m_eps );
		value_type const gamma_c = m_gamma.data()[c];
		value_type const mean_c  = m_running_mean.data()[c];

		value_type sum_dy      = static_cast<value_type>( 0 );
		value_type sum_dy_xhat = static_cast<value_type>( 0 );
		for ( std::size_t b = 0; b < in_shape[0]; ++b ) {
			for ( std::size_t i = 0; i < in_strides[1]; ++i ) {
				std::size_t const idx = b * in_strides[0] + c * in_strides[1] + i;
				value_type const xhat = ( input_data[idx] - mean_c ) * inv_std;
				sum_dy      += dy_data[idx];
				sum_dy_xhat += dy_data[idx] * xhat;
			}
		}

		value_type const scale = gamma_c * inv_std / static_cast<value_type>( M );
		for ( std::size_t b = 0; b < in_shape[0]; ++b ) {
			for ( std::size_t i = 0; i < in_strides[1]; ++i ) {
				std::size_t const idx = b * out_strides[0] + c * out_strides[1] + i;
				value_type const xhat = ( input_data[idx] - mean_c ) * inv_std;
				grad_output_data[idx] = scale * ( static_cast<value_type>( M ) * dy_data[idx] - sum_dy - xhat * sum_dy_xhat );
			}
		}
	}

	return this->getGradOutput();
}

} // namespace neural

#endif
