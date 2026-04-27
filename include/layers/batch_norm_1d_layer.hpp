#ifndef BATCH_NORM_1D_LAYER_HPP
#define BATCH_NORM_1D_LAYER_HPP

#include "layer_base.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>

#if NEURAL_CUDNN_ENABLED
#include "cudnn_handle.hpp"
#include "neural_cuda_kernels.hpp"
#include "neural_cuda_layer_sync.hpp"
#include <cudnn.h>
#endif

namespace neural {

#if NEURAL_CUDNN_ENABLED
namespace detail {
/// RAII wrapper for cudnnTensorDescriptor_t (local to BatchNorm1dLayer machinery).
struct Bn1dTensorDesc
{
	cudnnTensorDescriptor_t desc = nullptr;
	Bn1dTensorDesc() { cudnnCreateTensorDescriptor( &desc ); }
	~Bn1dTensorDesc()
	{
		if ( desc )
			cudnnDestroyTensorDescriptor( desc );
	}
	Bn1dTensorDesc( Bn1dTensorDesc const & )            = delete;
	Bn1dTensorDesc &operator=( Bn1dTensorDesc const & ) = delete;
};
} // namespace detail
#endif // NEURAL_CUDNN_ENABLED

namespace detail {

/// CPU forward for (N × C) batch normalization.
/// \p running_mean / \p running_var receive the batch statistics (used by backward).
/// \p last_running_mean / \p last_running_var hold and update the EMA inference stats.
template <typename T>
void bn1d_forward_cpu( T const *input, std::size_t N, std::size_t C, T const *gamma,
                       T const *beta, T *running_mean, T *running_var, T *last_running_mean,
                       T *last_running_var, T eps, T momentum, bool training,
                       bool is_first_forward, T *output )
{
	if ( training || is_first_forward ) {
		for ( std::size_t c = 0; c < C; ++c ) {
			T s = T{};
			for ( std::size_t n = 0; n < N; ++n )
				s += input[n * C + c];
			running_mean[c] = s / static_cast<T>( N );
		}
		for ( std::size_t c = 0; c < C; ++c ) {
			T v = T{};
			T const mc = running_mean[c];
			for ( std::size_t n = 0; n < N; ++n ) {
				T const d = input[n * C + c] - mc;
				v += d * d;
			}
			running_var[c] = v / static_cast<T>( N );
		}
	} else {
		for ( std::size_t c = 0; c < C; ++c ) {
			running_mean[c] = last_running_mean[c];
			running_var[c]  = last_running_var[c];
		}
	}

	for ( std::size_t n = 0; n < N; ++n ) {
		for ( std::size_t c = 0; c < C; ++c ) {
			T const inv_std = T{1} / std::sqrt( running_var[c] + eps );
			output[n * C + c] =
			    gamma[c] * ( input[n * C + c] - running_mean[c] ) * inv_std + beta[c];
		}
	}

	if ( training ) {
		for ( std::size_t c = 0; c < C; ++c ) {
			last_running_mean[c] =
			    last_running_mean[c] * ( T{1} - momentum ) + running_mean[c] * momentum;
			last_running_var[c] =
			    last_running_var[c] * ( T{1} - momentum ) + running_var[c] * momentum;
		}
	}
}

/// CPU backward for (N × C) batch normalization.
/// Uses the batch mean / variance stored in \p running_mean / \p running_var by the forward pass.
template <typename T>
void bn1d_backward_cpu( T const *dy, T const *input, std::size_t N, std::size_t C,
                        T const *gamma, T const *running_mean, T const *running_var, T eps,
                        T *grad_gamma, T *grad_beta, T *dx )
{
	for ( std::size_t c = 0; c < C; ++c ) {
		T const mc      = running_mean[c];
		T const inv_std = T{1} / std::sqrt( running_var[c] + eps );
		T sum_dy        = T{};
		T sum_dy_xhat   = T{};
		for ( std::size_t n = 0; n < N; ++n ) {
			T const xhat = ( input[n * C + c] - mc ) * inv_std;
			T const dyn  = dy[n * C + c];
			sum_dy      += dyn;
			sum_dy_xhat += dyn * xhat;
		}
		grad_beta[c]  = sum_dy;
		grad_gamma[c] = sum_dy_xhat;

		T const scale = gamma[c] * inv_std / static_cast<T>( N );
		for ( std::size_t n = 0; n < N; ++n ) {
			T const xhat  = ( input[n * C + c] - mc ) * inv_std;
			dx[n * C + c] = scale * ( static_cast<T>( N ) * dy[n * C + c] - sum_dy -
			                          xhat * sum_dy_xhat );
		}
	}
}

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────

/// Per-feature batch normalization for 2-D (\f$N \times C\f$) tensors.
///
/// Normalizes over the \f$N\f$ (batch) axis independently for each of the \f$C\f$ features
/// — the 1-D analogue of `BatchNormLayer` (NCHW) for use after fully-connected layers.
///
/// Learnable parameters: scale \f$\gamma\f$ and shift \f$\beta\f$, both of shape \f$(1, C)\f$.
/// Running statistics are accumulated with an EMA and used at evaluation time.
///
/// Both `Tensor<T>` (CPU) and `CudaTensor<T>` (GPU via cuDNN) are supported; the dispatch
/// is done at compile-time with `if constexpr`.
template <typename Tensor2D>
class BatchNorm1dLayer : public LayerBase<Tensor2D>
{
public:
	using tensor_t   = Tensor2D;
	using value_type = typename Tensor2D::value_type;
	using bias_t = Tensor2D;

	/// \p num_features = \f$C\f$.
	/// \p eps  avoids division by zero (default 1e-5).
	/// \p momentum matches PyTorch / cuDNN (default 0.1): \(\text{running} \leftarrow (1-m)\,\text{running} + m\,\text{batch}\).
	BatchNorm1dLayer( std::size_t num_features, value_type eps = 1e-5f, value_type momentum = 0.1f );

	std::size_t numFeatures() const noexcept { return m_num_features; }
	/// Batch-norm parameters must not be L2-regularized.
	bool nonRegularizable() const noexcept { return true; }

	void setTraining( bool training ) { m_training = training; }
	bool isTraining() const { return m_training; }

	/// Scale \f$\gamma\f$ (per-feature), shape \f$(1, C)\f$.
	Tensor2D       &getGamma()       { return m_gamma; }
	Tensor2D const &getGamma() const { return m_gamma; }
	/// Shift \f$\beta\f$ (per-feature), shape \f$(1, C)\f$.
	Tensor2D       &getBeta()        { return m_beta; }
	Tensor2D const &getBeta()  const { return m_beta; }

	// ── Optimizer interface (γ → weights, β → bias) ─────────────────────────
	Tensor2D       &getWeights()           { return m_gamma; }
	Tensor2D const &getWeights()     const { return m_gamma; }
	Tensor2D       &getGradWeights()       { return m_grad_gamma; }
	Tensor2D const &getGradWeights() const { return m_grad_gamma; }
	Tensor2D       &getBias()              { return m_beta; }
	Tensor2D const &getBias()        const { return m_beta; }
	Tensor2D       &getGradBias()          { return m_grad_beta; }
	Tensor2D const &getGradBias()    const { return m_grad_beta; }

	Tensor2D *forward();
	Tensor2D *backward();

private:
	std::size_t m_num_features;
	value_type  m_eps;
	value_type  m_momentum;
	bool        m_training = true;

	Tensor2D m_gamma;      ///< scale γ, shape (1, C), init 1
	Tensor2D m_beta;       ///< shift β, shape (1, C), init 0
	Tensor2D m_grad_gamma;
	Tensor2D m_grad_beta;

	/// Batch mean / variance from the latest training forward — consumed by backward.
	Tensor2D m_running_mean;
	Tensor2D m_running_var;
	/// EMA mean / variance — used for inference forward.
	Tensor2D m_last_running_mean;
	Tensor2D m_last_running_var;
};

// ─────────────────────────────────────────────────────────────────────────────

template <typename Tensor2D>
BatchNorm1dLayer<Tensor2D>::BatchNorm1dLayer( std::size_t num_features, value_type eps,
                                              value_type momentum )
    : m_num_features( num_features )
    , m_eps( eps )
    , m_momentum( momentum )
{}

template <typename Tensor2D>
Tensor2D *BatchNorm1dLayer<Tensor2D>::forward()
{
	Tensor2D const *input  = this->getInput();
	Tensor2D       *output = this->getOutput();

	std::size_t const N = input->rows();
	std::size_t const C = input->cols();

	bool is_first_forward = false;
	if ( m_gamma.cols() != C ) {
		m_gamma             = Tensor2D( 1, C, value_type{ 1 } );
		m_beta              = Tensor2D( 1, C, value_type{ 0 } );
		m_grad_gamma        = Tensor2D( 1, C, value_type{ 0 } );
		m_grad_beta         = Tensor2D( 1, C, value_type{ 0 } );
		m_running_mean      = Tensor2D( 1, C, value_type{ 0 } );
		m_running_var       = Tensor2D( 1, C, value_type{ 0 } );
		m_last_running_mean = Tensor2D( 1, C, value_type{ 0 } );
		m_last_running_var  = Tensor2D( 1, C, value_type{ 1 } );
		is_first_forward    = true;
	}
	if ( output->rows() != N || output->cols() != C )
		*output = Tensor2D( N, C );

	// Compile-time dispatch: CudaTensor has copyToHost(), Tensor (CPU) does not.
	if constexpr ( requires {
		               std::declval<Tensor2D &>().copyToHost( std::declval<value_type *>() );
	               } ) {
#if NEURAL_CUDNN_ENABLED
		cudnnHandle_t &handle = CudnnHandle::instance();

		detail::Bn1dTensorDesc x_desc;
		cudnnStatus_t st = cudnnSetTensor4dDescriptor(
		    x_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, static_cast<int>( N ),
		    static_cast<int>( C ), 1, 1 );
		if ( st != CUDNN_STATUS_SUCCESS )
			throw std::runtime_error(
			    "BatchNorm1dLayer::forward: cudnnSetTensor4dDescriptor (x) failed" );

		detail::Bn1dTensorDesc y_desc;
		st = cudnnSetTensor4dDescriptor( y_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		                                 static_cast<int>( N ), static_cast<int>( C ), 1, 1 );
		if ( st != CUDNN_STATUS_SUCCESS )
			throw std::runtime_error(
			    "BatchNorm1dLayer::forward: cudnnSetTensor4dDescriptor (y) failed" );

		detail::Bn1dTensorDesc bn_desc;
		st = cudnnDeriveBNTensorDescriptor( bn_desc.desc, x_desc.desc, CUDNN_BATCHNORM_SPATIAL );
		if ( st != CUDNN_STATUS_SUCCESS )
			throw std::runtime_error(
			    "BatchNorm1dLayer::forward: cudnnDeriveBNTensorDescriptor failed" );

		float const alpha    = 1.f;
		float const beta_val = 0.f;
		if ( m_training || is_first_forward ) {
			Tensor2D saved_inv_var( 1, C, value_type{ 0 } );
			double const exp_avg = static_cast<double>( m_momentum );
			st = cudnnBatchNormalizationForwardTraining(
			    handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta_val, x_desc.desc, input->data(),
			    y_desc.desc, output->data(), bn_desc.desc, m_gamma.data(), m_beta.data(),
			    exp_avg, m_last_running_mean.data(), m_last_running_var.data(),
			    static_cast<double>( m_eps ), m_running_mean.data(), saved_inv_var.data() );
			if ( st != CUDNN_STATUS_SUCCESS )
				throw std::runtime_error(
				    "BatchNorm1dLayer::forward: cudnnBatchNormalizationForwardTraining failed" );

			// Convert cuDNN's saved_inv_variance → actual variance for use in backward.
			cudaError_t const rv_err = cuda_bn_running_var_from_saved_inv_std_float(
			    m_running_var.data(), saved_inv_var.data(), C, static_cast<float>( m_eps ) );
			if ( rv_err != cudaSuccess )
				throw std::runtime_error( "BatchNorm1dLayer::forward: "
				                          "cuda_bn_running_var_from_saved_inv_std_float failed" );
		} else {
			st = cudnnBatchNormalizationForwardInference(
			    handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta_val, x_desc.desc, input->data(),
			    y_desc.desc, output->data(), bn_desc.desc, m_gamma.data(), m_beta.data(),
			    m_last_running_mean.data(), m_last_running_var.data(),
			    static_cast<double>( m_eps ) );
			if ( st != CUDNN_STATUS_SUCCESS )
				throw std::runtime_error( "BatchNorm1dLayer::forward: "
				                          "cudnnBatchNormalizationForwardInference failed" );
			m_running_mean = m_last_running_mean;
			m_running_var  = m_last_running_var;
		}
#else
		throw std::runtime_error( "BatchNorm1dLayer: CUDA/cuDNN not available in this build" );
#endif
#if NEURAL_CUDNN_ENABLED
		cuda_layer_sync();
#endif
	} else {
		// ── CPU path ─────────────────────────────────────────────────────────────
		detail::bn1d_forward_cpu(
		    input->data(), N, C, m_gamma.data(), m_beta.data(), m_running_mean.data(),
		    m_running_var.data(), m_last_running_mean.data(), m_last_running_var.data(),
		    m_eps, m_momentum, m_training, is_first_forward, output->data() );
	}

	return output;
}

template <typename Tensor2D>
Tensor2D *BatchNorm1dLayer<Tensor2D>::backward()
{
	Tensor2D const *grad_in  = this->getGradInput();  // dL/dy
	Tensor2D       *grad_out = this->getGradOutput(); // dL/dx

	std::size_t const N = grad_in->rows();
	std::size_t const C = grad_in->cols();

	if ( grad_out->rows() != N || grad_out->cols() != C )
		*grad_out = Tensor2D( N, C );
	if ( m_grad_gamma.cols() != C )
		m_grad_gamma = Tensor2D( 1, C, value_type{ 0 } );
	if ( m_grad_beta.cols() != C )
		m_grad_beta = Tensor2D( 1, C, value_type{ 0 } );

	if constexpr ( requires {
		               std::declval<Tensor2D &>().copyToHost( std::declval<value_type *>() );
	               } ) {
#if NEURAL_CUDNN_ENABLED
		// Same (N, C) ↔ (N, C, 1, 1) NCHW view as in forward; see comment there.
		// Reconstruct saved_inv_variance from the batch variance stored during forward.
		Tensor2D saved_inv_var( 1, C, value_type{ 0 } );
		cudaError_t const si_err = cuda_bn_saved_inv_std_from_variance_float(
		    saved_inv_var.data(), m_running_var.data(), C, static_cast<float>( m_eps ) );
		if ( si_err != cudaSuccess )
			throw std::runtime_error( "BatchNorm1dLayer::backward: "
			                          "cuda_bn_saved_inv_std_from_variance_float failed" );

		cudnnHandle_t &handle = CudnnHandle::instance();

		detail::Bn1dTensorDesc x_desc;
		cudnnStatus_t st = cudnnSetTensor4dDescriptor(
		    x_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, static_cast<int>( N ),
		    static_cast<int>( C ), 1, 1 );
		if ( st != CUDNN_STATUS_SUCCESS )
			throw std::runtime_error(
			    "BatchNorm1dLayer::backward: cudnnSetTensor4dDescriptor (x) failed" );

		detail::Bn1dTensorDesc dy_desc;
		st = cudnnSetTensor4dDescriptor( dy_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		                                 static_cast<int>( N ), static_cast<int>( C ), 1, 1 );
		if ( st != CUDNN_STATUS_SUCCESS )
			throw std::runtime_error(
			    "BatchNorm1dLayer::backward: cudnnSetTensor4dDescriptor (dy) failed" );

		detail::Bn1dTensorDesc dx_desc;
		st = cudnnSetTensor4dDescriptor( dx_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		                                 static_cast<int>( N ), static_cast<int>( C ), 1, 1 );
		if ( st != CUDNN_STATUS_SUCCESS )
			throw std::runtime_error(
			    "BatchNorm1dLayer::backward: cudnnSetTensor4dDescriptor (dx) failed" );

		detail::Bn1dTensorDesc bn_desc;
		st = cudnnDeriveBNTensorDescriptor( bn_desc.desc, x_desc.desc, CUDNN_BATCHNORM_SPATIAL );
		if ( st != CUDNN_STATUS_SUCCESS )
			throw std::runtime_error(
			    "BatchNorm1dLayer::backward: cudnnDeriveBNTensorDescriptor failed" );

		float const alpha_data  = 1.f, beta_data  = 0.f;
		float const alpha_param = 1.f, beta_param = 0.f;
		st = cudnnBatchNormalizationBackward(
		    handle, CUDNN_BATCHNORM_SPATIAL, &alpha_data, &beta_data, &alpha_param, &beta_param,
		    x_desc.desc, this->getInput()->data(), dy_desc.desc, grad_in->data(), dx_desc.desc,
		    grad_out->data(), bn_desc.desc, m_gamma.data(), m_grad_gamma.data(),
		    m_grad_beta.data(), static_cast<double>( m_eps ), m_running_mean.data(),
		    saved_inv_var.data() );
		if ( st != CUDNN_STATUS_SUCCESS )
			throw std::runtime_error(
			    "BatchNorm1dLayer::backward: cudnnBatchNormalizationBackward failed" );
#else
		throw std::runtime_error( "BatchNorm1dLayer: CUDA/cuDNN not available in this build" );
#endif
#if NEURAL_CUDNN_ENABLED
		cuda_layer_sync();
#endif
	} else {
		// ── CPU path ─────────────────────────────────────────────────────────────
		detail::bn1d_backward_cpu(
		    grad_in->data(), this->getInput()->data(), N, C, m_gamma.data(),
		    m_running_mean.data(), m_running_var.data(), m_eps, m_grad_gamma.data(),
		    m_grad_beta.data(), grad_out->data() );
	}

	return grad_out;
}

} // namespace neural

#endif // BATCH_NORM_1D_LAYER_HPP
