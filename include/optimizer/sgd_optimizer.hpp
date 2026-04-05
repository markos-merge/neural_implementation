#ifndef SGD_OPTIMIZER_HPP
#define SGD_OPTIMIZER_HPP

#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <functional>
#include <type_traits>
#include "layer.hpp"

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

		typename Tensor::value_type m_learning_rate = static_cast<typename Tensor::value_type>(0.01);
		std::size_t m_batch_size = 50;
		std::size_t m_epochs = 100;

		void train( std::vector< Tensor > &inputs, std::vector< Tensor > &targets,
		           ProgressCallback callback = nullptr );
	private:
		void initialize();
		void applyStep();
	private:
		NN &m_nn;

};

namespace detail {

template < typename Tensor, typename Layer >
void updateParam_impl( Tensor &param, Tensor const &grad, typename Tensor::value_type learning_rate )
{
	param.mulNSubstractInPlace( grad, learning_rate );
}

template< typename Tensor, typename Layer >
void updateLayer_impl( Layer &layer, typename Tensor::value_type learning_rate )
{
	if constexpr ( UpdateableLayer< std::decay_t<Layer>, Tensor> ) {
		updateParam_impl<typename std::decay_t<Layer>::tensor_t, typename std::decay_t<Layer>>( layer.getWeights(), layer.getGradWeights(), learning_rate );
		updateParam_impl<typename std::decay_t<Layer>::bias_t, typename std::decay_t<Layer>>( layer.getBias(), layer.getGradBias(), learning_rate );
	} else if constexpr ( requires( Layer &l ) { l.forEachLayer( []( auto && ) {} ); } ) {
		layer.forEachLayer( [&]( auto &&sub ) {
			detail::updateLayer_impl< typename std::decay_t<decltype(sub)>::tensor_t, std::decay_t<decltype(sub)> >( sub, learning_rate );
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
		detail::updateLayer_impl<Tensor>( layer, this->m_learning_rate );
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

namespace detail {
/// Copy \p num_rows samples starting at \p index_offset_in_epoch into \p tensor rows 0..num_rows-1.
/// \p index_offset_in_epoch is the position in the shuffled \p batch_indices (same as the outer
/// batch loop variable, e.g. 0, batch_size, 2*batch_size, ...).
template <typename Tensor>
void copy_batch_to_tensor( Tensor &tensor, std::vector<Tensor> &inputs,
                           std::vector<std::size_t> const &batch_indices,
                           std::size_t index_offset_in_epoch, std::size_t num_rows )
{
	for ( std::size_t r = 0; r < num_rows; ++r ) {
		tensor.assignTensorAsRow( r, inputs[batch_indices[index_offset_in_epoch + r]] );
	}
}
	
template <typename Tensor>
void copy_batch_to_tensor_block( Tensor &tensor, std::vector<Tensor> &inputs )
{
	for ( std::size_t r = 0; r < inputs.size(); ++r ) {
		tensor.assignTensorAsRow( r, inputs[r] );
	}
}

} // namespace detail

template< typename Tensor_t, typename NN >
void SGDOptimizer<Tensor_t, NN>::train( std::vector< Tensor_t > &inputs,
                                     std::vector< Tensor_t > &targets,
                                     typename SGDOptimizer<Tensor_t, NN>::ProgressCallback callback )
{
	initialize();
	// Only full batches so buffer shapes stay fixed (remainder samples are skipped each epoch).
	std::size_t const total_batches = inputs.size() / m_batch_size;
	if ( total_batches == 0 ) {
		return;
	}
	std::vector< int > batch_indices_int( inputs.size() );

	Tensor_t inputs_tensor( inputs.size(), inputs[0].cols() );
	Tensor_t targets_tensor( targets.size(), targets[0].cols() );

	detail::copy_batch_to_tensor_block( inputs_tensor, inputs );
	detail::copy_batch_to_tensor_block( targets_tensor, targets );

	for( std::size_t epoch = 0; epoch < this->m_epochs; ++epoch ) {
		bool stop_training = false;
		std::vector< std::size_t > batch_indices( inputs.size() );
		std::iota( batch_indices.begin(), batch_indices.end(), 0 );
		std::shuffle( batch_indices.begin(), batch_indices.end(),
		              std::mt19937( static_cast<std::mt19937::result_type>( epoch ) ) );
		for ( std::size_t r = 0; r < batch_indices.size(); ++r ) {
			batch_indices_int[static_cast<std::size_t>( r )] =
			    static_cast<int>( batch_indices[static_cast<std::size_t>( r )] );
		}

		std::size_t batch_idx = 0;
		typename Tensor_t::value_type loss = 0.0;

		for( std::size_t i = 0; i < total_batches * m_batch_size; i += m_batch_size ) {
			m_nn.ensureBuffersForShape( m_batch_size, inputs[0].cols() );
			m_nn.inputLatch().input()->assignTensorBlock( inputs_tensor, batch_indices_int, i,
			                                              m_batch_size );
			m_nn.targetBuffer().assignTensorBlock( targets_tensor, batch_indices_int, i, m_batch_size );
			loss = m_nn.trainStep();

			applyStep();

			if ( callback && callback( epoch, m_epochs, batch_idx, total_batches, loss ) ) {
				stop_training = true;
				break;
			}
			++batch_idx;
		}
		if ( stop_training ) {
			break;
		}
	}
}


} // namespace neural
// 
#endif
