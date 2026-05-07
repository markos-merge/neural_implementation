#ifndef REFACTOR_BATCH_NORM_LAYER_HPP
#define REFACTOR_BATCH_NORM_LAYER_HPP

#include <array>
#include <cstddef>
#include <stdexcept>
#include "layer_base.hpp"
#include "tensor_n.hpp"
#include <type_traits>
#include <vector>
#if NEURAL_CUDA_ENABLED
#include "cuda_tensor_n.hpp"
#include "neural_cuda_layer_sync.hpp"
#endif

namespace neural {
namespace refactor {

inline std::array<std::size_t, 4> batch_norm_input_as_nchw( std::vector<std::size_t> const &latch_shape )
{
	if ( latch_shape.size() == 2 ) {
		return { latch_shape[0], latch_shape[1], 1, 1 };
	}
	if ( latch_shape.size() == 4 ) {
		return { latch_shape[0], latch_shape[1], latch_shape[2], latch_shape[3] };
	}
	throw std::invalid_argument(
	    "BatchNormLayer: input latch must be rank 2 (N,C) or rank 4 (N,C,H,W)" );
}

inline std::vector<std::size_t> batch_norm_nchw_as_shape_vec( std::array<std::size_t, 4> const &nchw )
{
	return std::vector<std::size_t>( nchw.begin(), nchw.end() );
}

inline std::size_t batch_norm_channel_count_nchw( std::array<std::size_t, 4> const &nchw )
{
	return nchw[1];
}

template <typename T, typename Device = Cpu>
class BatchNormLayer : public LayerBase<T, Device>
{
public:
	explicit BatchNormLayer( T eps = T( 1e-5 ), T momentum = T( 0.1 ) );

	std::size_t numFeatures() const noexcept;

	bool nonRegularizable() const noexcept;

	void setTraining( bool training ) noexcept { m_training = training; }
	bool isTraining() const noexcept { return m_training; }

	T *getWeights() noexcept { return m_gamma.data(); }
	T *getGradWeights() noexcept { return m_grad_gamma.data(); }
	T *getBias() noexcept { return m_beta.data(); }
	T *getGradBias() noexcept { return m_grad_beta.data(); }

	std::size_t numWeightParams() const noexcept { return m_gamma.size(); }
	std::size_t numBiasParams() const noexcept { return m_beta.size(); }

	void forward();
	void backward();

private:
	using tensor_t = std::conditional_t<std::is_same_v<Device, Cuda>, CudaTensor4<T>, TensorN<4, T>>;

	T      m_eps;
	T      m_momentum;
	bool   m_training = true;

	tensor_t m_gamma;
	tensor_t m_beta;
	tensor_t m_grad_gamma;
	tensor_t m_grad_beta;

	tensor_t m_running_mean;
	tensor_t m_running_var;
	tensor_t m_last_running_mean;
	tensor_t m_last_running_var;
};

template <typename T, typename Device>
BatchNormLayer<T, Device>::BatchNormLayer( T eps, T momentum )
	: m_eps( eps )
	, m_momentum( momentum )
{
}

template <typename T, typename Device>
std::size_t BatchNormLayer<T, Device>::numFeatures() const noexcept
{
	return m_gamma.size() == 0 ? 0 : m_gamma.shape()[1];
}

template <typename T, typename Device>
bool BatchNormLayer<T, Device>::nonRegularizable() const noexcept
{
	return true;
}

template <typename T, typename Device>
void BatchNormLayer<T, Device>::forward()
{
	using value_type = typename tensor_t::value_type;

	LatchBase<T, Device> *in_latch  = this->getInput();
	LatchBase<T, Device> *out_latch = this->getOutput();

	std::array<std::size_t, 4> const nchw = batch_norm_input_as_nchw( in_latch->shape() );
	std::size_t const C                   = batch_norm_channel_count_nchw( nchw );
	bool                                   is_first_forward = false;

	if ( m_running_mean.size() != C ) {
		std::array<std::size_t, 4> const mean_shape{ 1, C, 1, 1 };
		m_running_mean      = tensor_t( mean_shape );
		m_last_running_mean = tensor_t( mean_shape );
	}
	if ( m_running_var.size() != C ) {
		std::array<std::size_t, 4> const var_shape{ 1, C, 1, 1 };
		m_running_var           = tensor_t( var_shape );
		m_last_running_var      = tensor_t( var_shape, static_cast<value_type>( 1 ) );
	}
	if ( m_gamma.size() != C ) {
		std::array<std::size_t, 4> const gamma_shape{ 1, C, 1, 1 };
		m_gamma = tensor_t( gamma_shape, static_cast<value_type>( 1 ) );
		is_first_forward = true;
	}
	if ( m_beta.size() != C ) {
		std::array<std::size_t, 4> const beta_shape{ 1, C, 1, 1 };
		m_beta               = tensor_t( beta_shape, static_cast<value_type>( 0 ) );
		is_first_forward     = true;
	}

	std::vector<std::size_t> const in_logical = in_latch->shape();
	if ( out_latch->shape() != in_logical ) {
		out_latch->resize( in_logical );
	}

	std::vector<std::size_t> const geom = batch_norm_nchw_as_shape_vec( nchw );
	tensor_t input_view( in_latch->fwdData(), geom, false );
	tensor_t output_view( out_latch->fwdData(), geom, false );

	batch_norm_forward_nchw( input_view, m_gamma, m_beta, m_running_mean, m_running_var,
	                         m_last_running_mean, m_last_running_var, m_eps, m_momentum,
	                         m_training, is_first_forward, output_view );

#if NEURAL_CUDA_ENABLED
	if constexpr ( std::is_same_v<Device, Cuda> ) {
		cuda_layer_sync();
	}
#endif
}

template <typename T, typename Device>
void BatchNormLayer<T, Device>::backward()
{
	LatchBase<T, Device> *g_in_latch  = this->getGradInput();
	LatchBase<T, Device> *g_out_latch = this->getGradOutput();
	LatchBase<T, Device> *in_latch    = this->getInput();

	std::vector<std::size_t> const g_logical = g_in_latch->shape();
	if ( g_out_latch->shape() != g_logical ) {
		g_out_latch->resize( g_logical );
	}

	std::array<std::size_t, 4> const nchw = batch_norm_input_as_nchw( g_logical );
	std::size_t const C                  = batch_norm_channel_count_nchw( nchw );

	std::vector<std::size_t> const geom = batch_norm_nchw_as_shape_vec( nchw );

	if ( m_grad_beta.size() != C ) {
		m_grad_beta = tensor_t( std::array<std::size_t, 4>{ 1, C, 1, 1 } );
	}
	if ( m_grad_gamma.size() != C ) {
		m_grad_gamma = tensor_t( std::array<std::size_t, 4>{ 1, C, 1, 1 } );
	}

	tensor_t dy_view( g_in_latch->fwdData(), geom, false );
	tensor_t x_view( in_latch->fwdData(), geom, false );
	tensor_t dx_view( g_out_latch->fwdData(), geom, false );

	batch_norm_backward_nchw( dy_view, x_view, m_gamma, m_running_mean, m_running_var, m_eps,
	                          m_grad_gamma, m_grad_beta, dx_view );

#if NEURAL_CUDA_ENABLED
	if constexpr ( std::is_same_v<Device, Cuda> ) {
		cuda_layer_sync();
	}
#endif
}

} // namespace refactor
} // namespace neural

#endif // REFACTOR_BATCH_NORM_LAYER_HPP
