#ifndef SEQUENTIAL_NN_HPP
#define SEQUENTIAL_NN_HPP

#include "latch_layer.hpp"
#include "linear_layer.hpp"
#include "mse_loss.hpp"
#include "relu_layer.hpp"
#include "sigmoid_layer.hpp"
#include "softmax_layer.hpp"
#include "terminal_latch_layer.hpp"
#include "tensor_like.hpp"
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace neural {

template <typename Tensor, typename Loss, typename ... Layers>
class SequentialNN
{
	static_assert( TensorLike<Tensor>, "Tensor must be a TensorLike type" );
	public:
		using tensor_tuple_t = std::tuple<Layers...>;
	public:
		explicit SequentialNN( Layers... layers );

		Tensor forward( Tensor const &input );
		Tensor backward( Tensor const &grad_output );

		/// Resize slot buffers (and target buffer) for a batch of shape \p batch_rows × \p input_cols.
		void ensureBuffersForShape( std::size_t batch_rows, std::size_t input_cols );

		/// Loss targets for the current batch; same shape as the network output (last slot).
		Tensor &targetBuffer();

		/// Run forward + loss + backward using activations already in the input latch and
		/// targets in \ref targetBuffer.
		typename Tensor::value_type trainStep();

		typename Tensor::value_type trainStep( Tensor const &input, Tensor const &target );

		template < typename UnaryOp >
		void forEachLayer( UnaryOp &&op );

		/// Zip each top-level layer with the matching element of \p zip_tuple (same arity as \c Layers...).
		template < typename Tuple, typename F >
		void forEachLayerZip( Tuple &zip_tuple, F &&f );

		LatchLayer< Tensor > &inputLatch();
		LatchLayer< Tensor > const &inputLatch() const;
		LatchLayer< Tensor > &outputLatch();
		LatchLayer< Tensor > const &outputLatch() const;

		TerminalLatchLayer< Tensor > const &inputTerminalLatch() const;
		TerminalLatchLayer< Tensor > const &outputTerminalLatch() const;

	private:
		void ensure_buffers( Tensor const &input );
		void wire_layers();
		void runForwardFromInputSlot();
		typename Tensor::value_type runTrainStepFromMappedBuffers();

		template < std::size_t... Is >
		void backward_reverse_ptr( std::index_sequence<Is...> );

		std::tuple< Layers... > m_layers;
		Loss m_loss;

		std::vector<LatchLayer<Tensor>> m_slots;
		TerminalLatchLayer<Tensor> m_input_terminal;
		TerminalLatchLayer<Tensor> m_output_terminal;
		Tensor m_target;
		bool m_buffers_ready = false;
		std::vector<std::pair<std::size_t, std::size_t>> m_cached_slot_dims;
};

namespace detail {

inline bool slot_dims_equal( std::vector<std::pair<std::size_t, std::size_t>> const &a,
                             std::vector<std::pair<std::size_t, std::size_t>> const &b )
{
	if ( a.size() != b.size() ) {
		return false;
	}
	for ( std::size_t i = 0; i < a.size(); ++i ) {
		if ( a[i].first != b[i].first || a[i].second != b[i].second ) {
			return false;
		}
	}
	return true;
}

template <typename Tensor >
std::pair<std::size_t, std::size_t> next_slot_shape( LinearLayer<Tensor> const &l,
                                                     std::pair<std::size_t, std::size_t> const &in )
{
	return { in.first, l.outFeatures() };
}

template < typename Tensor >
std::pair<std::size_t, std::size_t> next_slot_shape( ReLULayer<Tensor> const &,
                                                     std::pair<std::size_t, std::size_t> const &in )
{
	return in;
}

template < typename Tensor >
std::pair<std::size_t, std::size_t> next_slot_shape( SigmoidLayer<Tensor> const &,
                                                     std::pair<std::size_t, std::size_t> const &in )
{
	return in;
}

template < typename Tensor >
std::pair<std::size_t, std::size_t> next_slot_shape( SoftmaxLayer<Tensor> const &,
                                                     std::pair<std::size_t, std::size_t> const &in )
{
	return in;
}

template < typename Tensor, typename... Layers >
std::vector<std::pair<std::size_t, std::size_t>> infer_slot_shapes( Tensor const &first_input,
                                                                    std::tuple<Layers...> const &layers )
{
	std::vector<std::pair<std::size_t, std::size_t>> dims;
	dims.push_back( { first_input.rows(), first_input.cols() } );
	std::apply(
	    [&]( Layers const &... lyr ) {
		    (void)( ( dims.push_back( next_slot_shape( lyr, dims.back() ) ), ... ) );
	    },
	    layers );
	return dims;
}

} // namespace detail
//
template < typename Tensor, typename Loss, typename ...Layers >
SequentialNN<Tensor, Loss, Layers...>::SequentialNN( Layers... layers )
	: m_layers( std::forward<Layers>( layers )... )
{
}

template < typename Tensor, typename Loss, typename ...Layers >
void SequentialNN<Tensor, Loss, Layers...>::ensure_buffers( Tensor const &input )
{
	std::vector<std::pair<std::size_t, std::size_t>> const dims = detail::infer_slot_shapes( input, m_layers );

	if ( m_buffers_ready && detail::slot_dims_equal( dims, m_cached_slot_dims ) ) {
		return;
	}

	m_slots.clear();
	m_slots.reserve( dims.size() );
	for ( auto const &d : dims ) {
		m_slots.emplace_back( d.first, d.second );
	}

	m_input_terminal.rebind( m_slots.front().input(), m_slots.front().grads() );
	m_output_terminal.rebind( m_slots.back().input(), m_slots.back().grads() );

	wire_layers();
	auto const out_rows = dims.back().first;
	auto const out_cols = dims.back().second;
	m_cached_slot_dims = std::move( dims );
	m_target = Tensor( out_rows, out_cols,
	                   static_cast<typename Tensor::value_type>( 0 ) );
	m_buffers_ready = true;
}

template < typename Tensor, typename Loss, typename ...Layers >
void SequentialNN<Tensor, Loss, Layers...>::wire_layers()
{
	std::apply(
	    [this]( auto &... lyr ) {
		    std::size_t i = 0;
		    (void)( ( lyr.setInputOutputTensors( m_slots[i].input(),
		                                          m_slots[i + 1].input() ),
		              lyr.setGradInputOutputTensors( m_slots[i + 1].grads(),
		                                             m_slots[i].grads() ),
		              ++i ),
		            ... );
	    },
	    m_layers );
}

template < typename Tensor, typename Loss, typename ...Layers >
void SequentialNN<Tensor, Loss, Layers...>::runForwardFromInputSlot()
{
	std::apply(
	    [&]( auto &... lyr ) {
		    (void)( ( lyr.forward(), ... ) );
	    },
	    m_layers );
}

template < typename Tensor, typename Loss, typename ...Layers >
Tensor SequentialNN<Tensor, Loss, Layers...>::forward( Tensor const &input )
{
	ensure_buffers( input );
	*m_slots.front().input() = input;
	runForwardFromInputSlot();
	return Tensor( *m_slots.back().input() );
}

template < typename Tensor, typename Loss, typename ...Layers >
template < std::size_t... Is >
void SequentialNN<Tensor, Loss, Layers...>::backward_reverse_ptr( std::index_sequence<Is...> )
{
	(void)( ( std::get<sizeof...( Layers ) - 1 - Is>( m_layers ).backward(), ... ) );
}

template < typename Tensor, typename Loss, typename ...Layers >
Tensor SequentialNN<Tensor, Loss, Layers...>::backward( Tensor const &grad_output )
{
	if ( !m_buffers_ready ) {
		throw std::logic_error( "SequentialNN::backward: call forward first" );
	}
	*m_slots.back().grads() = grad_output;
	backward_reverse_ptr( std::make_index_sequence<sizeof...( Layers )>{} );
	return Tensor( *m_slots.front().grads() );
}

template < typename Tensor, typename Loss, typename ...Layers >
void SequentialNN<Tensor, Loss, Layers...>::ensureBuffersForShape( std::size_t batch_rows,
                                                                   std::size_t input_cols )
{
	Tensor stub( batch_rows, input_cols );
	ensure_buffers( stub );
}

template < typename Tensor, typename Loss, typename ...Layers >
Tensor &SequentialNN<Tensor, Loss, Layers...>::targetBuffer()
{
	return m_target;
}

template < typename Tensor, typename Loss, typename ...Layers >
typename Tensor::value_type SequentialNN<Tensor, Loss, Layers...>::runTrainStepFromMappedBuffers()
{
	if ( !m_buffers_ready ) {
		throw std::logic_error( "SequentialNN::trainStep: buffers not ready; call ensureBuffersForShape or forward first" );
	}
	runForwardFromInputSlot();
	auto const loss_tensor = m_loss.forward( *m_slots.back().input(), m_target );
	backward( m_loss.backward() );
	return loss_tensor( 0, 0 );
}

template < typename Tensor, typename Loss, typename ...Layers >
typename Tensor::value_type SequentialNN<Tensor, Loss, Layers...>::trainStep()
{
	return runTrainStepFromMappedBuffers();
}

template < typename Tensor, typename Loss, typename ...Layers >
typename Tensor::value_type SequentialNN<Tensor, Loss, Layers...>::trainStep( Tensor const &input,
                                                                              Tensor const &target )
{
	ensure_buffers( input );
	*m_slots.front().input() = input;
	m_target = target;
	return runTrainStepFromMappedBuffers();
}

template < typename Tensor, typename Loss, typename ...Layers >
template < typename UnaryOp >
void SequentialNN<Tensor, Loss, Layers...>::forEachLayer( UnaryOp &&op )
{
	std::apply( [&op]( auto &&...args ) { (op( args ), ...); }, m_layers );
}

template < typename Tensor, typename Loss, typename ...Layers >
template < typename Tuple, typename F >
void SequentialNN<Tensor, Loss, Layers...>::forEachLayerZip( Tuple &zip_tuple, F &&f )
{
	std::apply(
	    [&f, &zip_tuple]( auto &...layer_args ) {
		    std::apply(
		        [&f, &layer_args...]( auto &...zip_args ) {
			        (void)( ( f( layer_args, zip_args ), ... ), 0 );
		        },
		        zip_tuple );
	    },
	    m_layers );
}

template < typename Tensor, typename Loss, typename ...Layers >
LatchLayer<Tensor> &SequentialNN<Tensor, Loss, Layers...>::inputLatch()
{
	if ( !m_buffers_ready || m_slots.empty() ) {
		throw std::logic_error( "SequentialNN::inputLatch: buffers not ready" );
	}
	return m_slots.front();
}

template < typename Tensor, typename Loss, typename ...Layers >
LatchLayer<Tensor> const &SequentialNN<Tensor, Loss, Layers...>::inputLatch() const
{
	if ( !m_buffers_ready || m_slots.empty() ) {
		throw std::logic_error( "SequentialNN::inputLatch: buffers not ready" );
	}
	return m_slots.front();
}

template < typename Tensor, typename Loss, typename ...Layers >
LatchLayer<Tensor> &SequentialNN<Tensor, Loss, Layers...>::outputLatch()
{
	if ( !m_buffers_ready || m_slots.empty() ) {
		throw std::logic_error( "SequentialNN::outputLatch: buffers not ready" );
	}
	return m_slots.back();
}

template < typename Tensor, typename Loss, typename ...Layers >
LatchLayer<Tensor> const &SequentialNN<Tensor, Loss, Layers...>::outputLatch() const
{
	if ( !m_buffers_ready || m_slots.empty() ) {
		throw std::logic_error( "SequentialNN::outputLatch: buffers not ready" );
	}
	return m_slots.back();
}

template < typename Tensor, typename Loss, typename ...Layers >
TerminalLatchLayer<Tensor> const &SequentialNN<Tensor, Loss, Layers...>::inputTerminalLatch() const
{
	return m_input_terminal;
}

template < typename Tensor, typename Loss, typename ...Layers >
TerminalLatchLayer<Tensor> const &SequentialNN<Tensor, Loss, Layers...>::outputTerminalLatch() const
{
	return m_output_terminal;
}

template < typename Tensor, typename ... Layers >
using SequentialNNMSE = SequentialNN<Tensor, MSELoss<Tensor>, Layers...>;

} // namespace neural

#endif
