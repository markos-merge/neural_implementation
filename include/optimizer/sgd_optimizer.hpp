#ifndef SGD_OPTIMIZER_HPP
#define SGD_OPTIMIZER_HPP

#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <functional>
#include <future>
#include <type_traits>
#include "layer.hpp"
#include "tensor.hpp"
#include "cuda_tensor.hpp"

namespace neural {

template <typename Tensor, typename NN>
class SGDOptimizer
{
	public:
		/// Return true to request early stop; false continues training.
		using ProgressCallback =
		    std::function<bool( std::size_t epoch, std::size_t epoch_max,
		                        std::size_t batch_idx, std::size_t batch_max,
		                        typename Tensor::value_type loss )>;

		SGDOptimizer( NN &nn );

		typename Tensor::value_type m_learning_rate = static_cast<typename Tensor::value_type>( 0.01 );
		/// \(\lambda\) for \(\frac{\lambda}{2}\|W\|^2\) on **weights only** (biases are not regularized—standard practice). If training stalls or diverges, lower \(\lambda\) or \c m_learning_rate.
		typename Tensor::value_type m_l2_regularizer = static_cast< typename Tensor::value_type>( 0. );
		std::size_t m_batch_size = 50;
		std::size_t m_epochs = 100;

		/// Dataset rows are **host** \c ::neural::Tensor<float>. Batches are gathered on the host, optional \p op runs there, then \ref assignBatch copies into \c Tensor_t staging buffers and into the network latch.
		void train( std::vector<::neural::Tensor<float>> &inputs,
		            std::vector<::neural::Tensor<float>> &targets,
		            ProgressCallback callback = nullptr );

		/// Same as \c train(inputs, targets, callback) but uses \p op after each row gather (\c op holds input/target column counts and batch row count; default is no-op).
		template <typename Op>
		void train( std::vector<::neural::Tensor<float>> &inputs,
		            std::vector<::neural::Tensor<float>> &targets,
		            ProgressCallback callback,
		            Op const &op );
	private:
		void initialize();
		void applyStep();
	private:
		NN &m_nn;

};

namespace detail {

template < typename Tensor, typename Layer >
void updateParam_impl( Tensor &param, Tensor const &grad, typename Tensor::value_type learning_rate, typename Tensor::value_type l2_regularizer )
{
	if( l2_regularizer > 0. ) {
		param.mulNSubstractInPlace( param, learning_rate*l2_regularizer );
	}
	param.mulNSubstractInPlace( grad, learning_rate );
}

template< typename Tensor, typename Layer >
void updateLayer_impl( Layer &layer, typename Tensor::value_type learning_rate, typename Tensor::value_type l2_regularizer )
{
	if constexpr ( UpdateableLayer< std::decay_t<Layer>, Tensor> ) {
		updateParam_impl<typename std::decay_t<Layer>::tensor_t, typename std::decay_t<Layer>>( layer.getWeights(), layer.getGradWeights(), learning_rate, l2_regularizer );
		updateParam_impl<typename std::decay_t<Layer>::bias_t, typename std::decay_t<Layer>>(
		    layer.getBias(), layer.getGradBias(), learning_rate,
		    static_cast<typename Tensor::value_type>( 0 ) );
	} else if constexpr ( requires( Layer &l ) { l.forEachLayer( []( auto && ) {} ); } ) {
		layer.forEachLayer( [&]( auto &&sub ) {
			detail::updateLayer_impl< typename std::decay_t<decltype(sub)>::tensor_t, std::decay_t<decltype(sub)> >( sub, learning_rate, l2_regularizer );
		} );
	} else {
		
	}
}
}// namespace detail

template <typename Tensor, typename NN>
SGDOptimizer<Tensor, NN>::SGDOptimizer( NN &nn )
	: m_nn( nn )
{
}

template <typename Tensor, typename NN>
void SGDOptimizer<Tensor, NN>::applyStep()
{
	m_nn.forEachLayer( [this]( auto &&layer ) {
		detail::updateLayer_impl<Tensor>( layer, this->m_learning_rate, this->m_l2_regularizer );
	} );
}

template< typename Tensor, typename NN >
void SGDOptimizer<Tensor, NN>::initialize()
{
	m_nn.forEachLayer( [this]( auto &&layer ) {
		if constexpr ( requires { layer.initialize(); } ) {
			layer.initialize();
		}
	} );
}

template< typename Tensor_t, typename NN >
void SGDOptimizer<Tensor_t, NN>::train( std::vector<::neural::Tensor<float>> &inputs,
                                        std::vector<::neural::Tensor<float>> &targets,
                                        typename SGDOptimizer<Tensor_t, NN>::ProgressCallback callback )
{
	std::size_t const total_batches = inputs.size() / m_batch_size;
	if ( total_batches == 0 ) {
		return;
	}
	std::size_t const in_cols  = inputs[0].cols();
	std::size_t const out_cols = targets[0].cols();
	DefaultAssignBatchOp<float> const op( in_cols, out_cols, m_batch_size );
	train( inputs, targets, callback, op );
}

template< typename Tensor_t, typename NN >
template <typename Op>
void SGDOptimizer<Tensor_t, NN>::train( std::vector<::neural::Tensor<float>> &inputs,
                                        std::vector<::neural::Tensor<float>> &targets,
                                        typename SGDOptimizer<Tensor_t, NN>::ProgressCallback callback,
                                        Op const &op )
{
	std::size_t const total_batches = inputs.size() / m_batch_size;
	if ( total_batches == 0 ) {
		return;
	}
	std::vector< int > batch_indices_int( inputs.size() );
	std::size_t const in_cols  = inputs[0].cols();
	std::size_t const out_cols = targets[0].cols();

	m_nn.ensureBuffersForShape( m_batch_size, in_cols );

	::neural::Tensor<float> host_x0( m_batch_size, in_cols );
	::neural::Tensor<float> host_y0( m_batch_size, out_cols );
	::neural::Tensor<float> host_x1( m_batch_size, in_cols );
	::neural::Tensor<float> host_y1( m_batch_size, out_cols );

	Tensor_t batch_x0( m_batch_size, in_cols );
	Tensor_t batch_y0( m_batch_size, out_cols );
	Tensor_t batch_x1( m_batch_size, in_cols );
	Tensor_t batch_y1( m_batch_size, out_cols );

	for ( std::size_t epoch = 0; epoch < m_epochs; ++epoch ) {
		bool stop_training = false;
		std::vector<std::size_t> batch_indices( inputs.size() );
		std::iota( batch_indices.begin(), batch_indices.end(), 0 );
		std::shuffle( batch_indices.begin(), batch_indices.end(),
		              std::mt19937( static_cast<std::mt19937::result_type>( epoch ) ) );
		for ( std::size_t r = 0; r < batch_indices.size(); ++r ) {
			batch_indices_int[static_cast<std::size_t>( r )] =
			    static_cast<int>( batch_indices[static_cast<std::size_t>( r )] );
		}

		assignBatch( batch_x0, batch_y0, inputs, targets, batch_indices_int, 0, m_batch_size,
		             host_x0, host_y0, op );

		std::future<void> prefetch;
		std::size_t batch_idx = 0;
		typename Tensor_t::value_type loss = 0.0;

		for ( std::size_t k = 0; k < total_batches; ++k ) {
			if ( k > 0 ) {
				prefetch.wait();
			}
			int const cur = static_cast<int>( k % 2 );
			Tensor_t &bx = cur == 0 ? batch_x0 : batch_x1;
			Tensor_t &by = cur == 0 ? batch_y0 : batch_y1;

			m_nn.inputLatch().input()->assign( bx.data(), bx.size() );
			m_nn.targetBuffer().assign( by.data(), by.size() );

			if ( k + 1 < total_batches ) {
				std::size_t const ni = ( k + 1 ) * m_batch_size;
				int const nxt = 1 - cur;
				prefetch = std::async( std::launch::async, [&, nxt, ni]() {
					if ( nxt == 0 ) {
						assignBatch( batch_x0, batch_y0, inputs, targets, batch_indices_int, ni,
						             m_batch_size, host_x0, host_y0, op );
					} else {
						assignBatch( batch_x1, batch_y1, inputs, targets, batch_indices_int, ni,
						             m_batch_size, host_x1, host_y1, op );
					}
				} );
			}

			loss = m_nn.trainStep();

			applyStep();

			if ( callback && callback( epoch, m_epochs, batch_idx, total_batches, loss ) ) {
				if ( prefetch.valid() ) {
					prefetch.wait();
				}
				stop_training = true;
				break;
			}
			++batch_idx;
		}
		if ( prefetch.valid() ) {
			prefetch.wait();
		}
		if ( stop_training ) {
			break;
		}
	}

	m_nn.ensureBuffersForShape( m_batch_size, in_cols );
}


} // namespace neural

#endif
