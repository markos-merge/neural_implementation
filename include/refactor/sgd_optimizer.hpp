#ifndef REFACTOR_SGD_OPTIMIZER_HPP
#define REFACTOR_SGD_OPTIMIZER_HPP

#include <cstddef>
#include <functional>
#include <type_traits>
#include "device.hpp"
#include "mse_loss.hpp"
#include "sequential_nn.hpp"
#include "tensor.hpp"
#if NEURAL_CUDA_ENABLED
#include "cuda_tensor.hpp"
#endif

namespace neural {
namespace refactor {

template <typename T, typename Device = Cpu>
class SGDOptimizer
{
public:
	using LossFn = std::function<T( T const *, T const *, std::size_t, std::size_t, T * )>;

	// Full training step: forward → loss + gradient → backward → weight update.
	// data: host [N * in_features], targets: host [N * out_features].
	// Returns the scalar loss for the batch.
	T step( SequentialNN2<T, Device> &nn, T const *data, std::size_t N,
	        std::size_t in_features, T const *targets );

	T      m_learning_rate;
	T      m_l2_regularizer = T( 0 );
	LossFn m_loss;

	explicit SGDOptimizer( T lr, LossFn loss = MSELoss<T, Device>{} );

private:
	using grad_t = std::conditional_t<std::is_same_v<Device, Cuda>, CudaTensor<T>, Tensor<T>>;

	/// Reused across `step()` calls to avoid per-batch device (or host) alloc/free.
	grad_t m_loss_grad_buf;

	void applyUpdate( T *param, T const *grad, std::size_t n, bool regularize ) const;
};

// ---------------------------------------------------------------------------

template <typename T, typename Device>
SGDOptimizer<T, Device>::SGDOptimizer( T lr, LossFn loss )
    : m_learning_rate( lr )
    , m_loss( std::move( loss ) )
{
}

template <typename T, typename Device>
void SGDOptimizer<T, Device>::applyUpdate( T *param, T const *grad, std::size_t n,
                                            bool regularize ) const
{
	for ( std::size_t i = 0; i < n; ++i ) {
		T delta = m_learning_rate * grad[i];
		if ( regularize && m_l2_regularizer > T( 0 ) )
			delta += m_learning_rate * m_l2_regularizer * param[i];
		param[i] -= delta;
	}
}

template <typename T, typename Device>
T SGDOptimizer<T, Device>::step( SequentialNN2<T, Device> &nn, T const *data,
                                  std::size_t N, std::size_t in_features,
                                  T const *targets )
{
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

#endif // REFACTOR_SGD_OPTIMIZER_HPP
