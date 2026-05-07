#ifndef REFACTOR_MOMENTUM_SGD_OPTIMIZER_HPP
#define REFACTOR_MOMENTUM_SGD_OPTIMIZER_HPP

#include <cstddef>
#include <functional>
#include <type_traits>
#include <vector>
#include "device.hpp"
#include "mse_loss.hpp"
#include "sequential_nn.hpp"
#include "tensor.hpp"
#if NEURAL_CUDA_ENABLED
#include "cuda_tensor.hpp"
#include "neural_cuda_runtime.hpp"
#endif

namespace neural {
namespace refactor {

// SGD with momentum.
// v_new = momentum * v_old + grad
// param -= lr * v_new  (+ optional L2 on weights)
// Velocity buffers are allocated on the first step and indexed positionally
// — forEachLayer always visits parameters in the same order.
template <typename T, typename Device = Cpu>
class MomentumSGDOptimizer
{
public:
	using LossFn = std::function<T( T const *, T const *, std::size_t, std::size_t, T * )>;

	// Full training step: forward → loss + gradient → backward → weight update.
	// data: host [N * in_features], targets: host [N * out_features].
	// Returns the scalar loss for the batch.
	T step( SequentialNN2<T, Device> &nn, T const *data, std::size_t N,
	        std::size_t in_features, T const *targets );

	T      m_learning_rate;
	T      m_momentum;
	T      m_l2_regularizer = T( 0 );
	LossFn m_loss;

	explicit MomentumSGDOptimizer( T lr, T momentum = T( 0.9 ), LossFn loss = MSELoss<T, Device>{} );

private:
	using grad_t = std::conditional_t<std::is_same_v<Device, Cuda>, CudaTensor<T>, Tensor<T>>;

	mutable std::vector<grad_t> m_velocity;
	mutable std::size_t         m_vel_idx = 0;

	/// Reused across `step()` calls to avoid per-batch device (or host) alloc/free.
	grad_t m_loss_grad_buf;

	void applyUpdate( T *param, T const *grad, std::size_t n, bool regularize ) const;
};

// ---------------------------------------------------------------------------

template <typename T, typename Device>
MomentumSGDOptimizer<T, Device>::MomentumSGDOptimizer( T lr, T momentum, LossFn loss )
    : m_learning_rate( lr )
    , m_momentum( momentum )
    , m_loss( std::move( loss ) )
{
}

template <typename T, typename Device>
void MomentumSGDOptimizer<T, Device>::applyUpdate( T *param, T const *grad, std::size_t n,
                                                    bool regularize ) const
{
	if ( m_vel_idx == m_velocity.size() )
		m_velocity.emplace_back( 1, n, T( 0 ) );
	grad_t &vel = m_velocity[m_vel_idx++];

	grad_t grad_view( const_cast<T *>( grad ), 1, n, false );
	grad_t param_view( param, 1, n, false );

	vel.geam( m_momentum, T( 1 ), grad_view );
	if ( regularize && m_l2_regularizer > T( 0 ) )
		vel.axpy( param_view, m_l2_regularizer );

	param_view.axpy( vel, -m_learning_rate );
}

template <typename T, typename Device>
T MomentumSGDOptimizer<T, Device>::step( SequentialNN2<T, Device> &nn, T const *data,
                                          std::size_t N, std::size_t in_features,
                                          T const *targets )
{
	m_vel_idx = 0;

	nn.forward( data, N, in_features );
	std::size_t const out_n = nn.outputSize();
	std::size_t const C     = out_n / N;
	if ( m_loss_grad_buf.rows() != N || m_loss_grad_buf.cols() != C )
		m_loss_grad_buf = grad_t( N, C );
	T const loss = m_loss( nn.output(), targets, N, C, m_loss_grad_buf.data() );

	nn.backward( m_loss_grad_buf.data() );

	nn.forEachLayer( [this]( auto &l ) {
		if constexpr ( requires { l.getWeights(); l.numWeightParams(); } ) {
			if ( T *w = l.getWeights(); w )
				if ( T const *dw = l.getGradWeights(); dw )
					applyUpdate( w, dw, l.numWeightParams(), true );
		}
		if constexpr ( requires { l.getBias(); l.numBiasParams(); } ) {
			if ( T *b = l.getBias(); b )
				if ( T const *db = l.getGradBias(); db )
					applyUpdate( b, db, l.numBiasParams(), false );
		}
	} );

	return loss;
}

} // namespace refactor
} // namespace neural

#endif // REFACTOR_MOMENTUM_SGD_OPTIMIZER_HPP
