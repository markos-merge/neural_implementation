#ifndef SGD_OPTIMIZER_HPP
#define SGD_OPTIMIZER_HPP

#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <functional>
#include "layer.hpp"

namespace neural {

template <typename Tensor, typename NN>
class SGDOptimizer
{
	public:
		using ProgressCallback =
		    std::function<void( std::size_t epoch, std::size_t epoch_max,
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
	param -= grad * learning_rate;
}

template< typename Tensor, typename Layer >
void updateLayer_impl( Layer &layer, typename Tensor::value_type learning_rate )
{
	if constexpr ( UpdateableLayer< std::decay_t<Layer>, Tensor> ) {
		updateParam_impl<Tensor, Layer>( layer.getWeights(), layer.getGradWeights(), learning_rate );
		updateParam_impl<Tensor, Layer>( layer.getBias(), layer.getGradBias(), learning_rate );
	} else {
		//nop
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
		if constexpr ( UpdateableLayer< std::decay_t<decltype( layer )>, Tensor> ) {
			layer.initialize();
		} else {
			//nop
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
} // namespace detail

template< typename Tensor, typename NN >
void SGDOptimizer<Tensor, NN>::train( std::vector< Tensor > &inputs,
                                     std::vector< Tensor > &targets,
                                     ProgressCallback callback )
{
	initialize();
	std::size_t const total_batches =
	    ( inputs.size() + m_batch_size - 1 ) / m_batch_size;

	for( std::size_t epoch = 0; epoch < this->m_epochs; ++epoch ) {
		std::vector< std::size_t > batch_indices( inputs.size() );
		std::iota( batch_indices.begin(), batch_indices.end(), 0 );
		std::shuffle( batch_indices.begin(), batch_indices.end(),
		              std::mt19937( static_cast<std::mt19937::result_type>( epoch ) ) );
		std::size_t batch_idx = 0;
		typename Tensor::value_type loss = 0.0;

		for( std::size_t i = 0; i < inputs.size(); i += m_batch_size ) {
			std::size_t const cur_batch_size = std::min( m_batch_size, inputs.size() - i );
			if ( cur_batch_size == 0 )
				continue;
			m_nn.ensureBuffersForShape( cur_batch_size, inputs[0].cols() );
			detail::copy_batch_to_tensor( *m_nn.inputLatch().input(), inputs,
			                              batch_indices, i, cur_batch_size );
			detail::copy_batch_to_tensor( m_nn.targetBuffer(), targets, batch_indices, i,
			                              cur_batch_size );
			loss = m_nn.trainStep();

			applyStep();

			if ( callback )
				callback( epoch, m_epochs, batch_idx, total_batches, loss );
			++batch_idx;
		}
	}
}


} // namespace neural
// 
#endif