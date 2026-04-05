#ifndef CONVOLUTIONAL_BOX_HPP
#define CONVOLUTIONAL_BOX_HPP

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include "layer_base.hpp"

namespace neural {

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
class ConvolutionalBox : public LayerBase<Tensor2D_t>
{
	static_assert( TensorN_t::rank_v == 4, "ConvolutionalBox requires a rank-4 TensorN" );
	public:
		using tensor_tuple_t = std::tuple<Layers...>;

		using value_type = typename Tensor2D_t::value_type;
	public:

		ConvolutionalBox( std::size_t channels, std::size_t height, std::size_t width,
		                  std::size_t flat_output_cols, Layers... layers );

		ConvolutionalBox( ConvolutionalBox const &other );
		ConvolutionalBox( ConvolutionalBox &&other ) noexcept;
		ConvolutionalBox &operator=( ConvolutionalBox const &other );
		ConvolutionalBox &operator=( ConvolutionalBox &&other ) noexcept;

		void initialize();
		void applyUpdate( value_type learning_rate );

		Tensor2D_t *forward();
		Tensor2D_t *backward();

		std::size_t outputCols() const { return m_flat_output_cols; }

		template <typename UnaryOp>
		void forEachLayer( UnaryOp &&op );

		/// Zip each inner layer with the matching element of \p zip_tuple (same arity as inner \c Layers...).
		template <typename Tuple, typename F>
		void forEachLayerZip( Tuple &zip_tuple, F &&f );

	private:
		void wire_layers();
		void ensure_buffers( std::size_t batch_size );

		template <std::size_t... Is>
		void backward_reverse( std::index_sequence<Is...> );

		std::size_t m_channels;
		std::size_t m_height;
		std::size_t m_width;
		std::size_t m_flat_output_cols;

		std::tuple<Layers...> m_layers;

		std::array<TensorN_t, sizeof...( Layers ) + 1> m_fwd_slots;
		std::array<TensorN_t, sizeof...( Layers ) + 1> m_bwd_slots;

		std::size_t m_cached_batch_size = 0;
};

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
ConvolutionalBox<TensorN_t, Tensor2D_t, Layers...>::ConvolutionalBox(
	std::size_t channels, std::size_t height, std::size_t width,
	std::size_t flat_output_cols, Layers... layers )
	: m_channels( channels )
	, m_height( height )
	, m_width( width )
	, m_flat_output_cols( flat_output_cols )
	, m_layers( std::forward<Layers>( layers )... )
{
	wire_layers();
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
ConvolutionalBox<TensorN_t, Tensor2D_t, Layers...>::ConvolutionalBox( ConvolutionalBox const &other )
	: LayerBase<Tensor2D_t>( other )
	, m_channels( other.m_channels )
	, m_height( other.m_height )
	, m_width( other.m_width )
	, m_flat_output_cols( other.m_flat_output_cols )
	, m_layers( other.m_layers )
	, m_fwd_slots( other.m_fwd_slots )
	, m_bwd_slots( other.m_bwd_slots )
	, m_cached_batch_size( other.m_cached_batch_size )
{
	wire_layers();
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
ConvolutionalBox<TensorN_t, Tensor2D_t, Layers...>::ConvolutionalBox( ConvolutionalBox &&other ) noexcept
	: LayerBase<Tensor2D_t>( std::move( other ) )
	, m_channels( other.m_channels )
	, m_height( other.m_height )
	, m_width( other.m_width )
	, m_flat_output_cols( other.m_flat_output_cols )
	, m_layers( std::move( other.m_layers ) )
	, m_fwd_slots( std::move( other.m_fwd_slots ) )
	, m_bwd_slots( std::move( other.m_bwd_slots ) )
	, m_cached_batch_size( other.m_cached_batch_size )
{
	wire_layers();
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
ConvolutionalBox<TensorN_t, Tensor2D_t, Layers...> &
ConvolutionalBox<TensorN_t, Tensor2D_t, Layers...>::operator=( ConvolutionalBox const &other )
{
	if ( this != &other ) {
		LayerBase<Tensor2D_t>::operator=( other );
		m_channels          = other.m_channels;
		m_height            = other.m_height;
		m_width             = other.m_width;
		m_flat_output_cols  = other.m_flat_output_cols;
		m_layers            = other.m_layers;
		m_fwd_slots         = other.m_fwd_slots;
		m_bwd_slots         = other.m_bwd_slots;
		m_cached_batch_size = other.m_cached_batch_size;
		wire_layers();
	}
	return *this;
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
ConvolutionalBox<TensorN_t, Tensor2D_t, Layers...> &
ConvolutionalBox<TensorN_t, Tensor2D_t, Layers...>::operator=( ConvolutionalBox &&other ) noexcept
{
	if ( this != &other ) {
		LayerBase<Tensor2D_t>::operator=( std::move( other ) );
		m_channels          = other.m_channels;
		m_height            = other.m_height;
		m_width             = other.m_width;
		m_flat_output_cols  = other.m_flat_output_cols;
		m_layers            = std::move( other.m_layers );
		m_fwd_slots         = std::move( other.m_fwd_slots );
		m_bwd_slots         = std::move( other.m_bwd_slots );
		m_cached_batch_size = other.m_cached_batch_size;
		wire_layers();
	}
	return *this;
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
void ConvolutionalBox<TensorN_t, Tensor2D_t, Layers...>::wire_layers()
{
	std::apply(
	    [this]( auto &...lyr ) {
		    std::size_t i = 0;
		    ( ( lyr.setInputOutputTensors( &m_fwd_slots[i], &m_fwd_slots[i + 1] ),
		        lyr.setGradInputOutputTensors( &m_bwd_slots[i + 1], &m_bwd_slots[i] ),
		        ++i ),
		      ... );
	    },
	    m_layers );
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
void ConvolutionalBox<TensorN_t, Tensor2D_t, Layers...>::ensure_buffers( std::size_t batch_size )
{
	if ( batch_size == m_cached_batch_size )
		return;
	m_fwd_slots[0] = TensorN_t( { batch_size, m_channels, m_height, m_width } );
	m_cached_batch_size = batch_size;
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
void ConvolutionalBox<TensorN_t, Tensor2D_t, Layers...>::initialize()
{
	std::apply(
	    []( auto &...lyr ) {
		    ( ..., ( [&lyr]() {
			    if constexpr ( requires { lyr.initialize(); } ) {
				    lyr.initialize();
			    }
		    }() ) );
	    },
	    m_layers );
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
void ConvolutionalBox<TensorN_t, Tensor2D_t, Layers...>::applyUpdate( value_type learning_rate )
{
	std::apply(
	    [learning_rate]( auto &...lyr ) {
		    ( ..., ( [&lyr, learning_rate]() {
			    if constexpr ( requires {
				    lyr.getWeights();
				    lyr.getBias();
				    lyr.getGradWeights();
				    lyr.getGradBias();
			    } ) {
				    auto &w        = lyr.getWeights();
				    auto const &gw = lyr.getGradWeights();
					#ifdef _OPENMP
					#pragma omp parallel for schedule( static )
					#endif
				    for ( std::size_t i = 0; i < w.size(); ++i )
					    w.data()[i] -= gw.data()[i] * learning_rate;

				    auto &b        = lyr.getBias();
				    auto const &gb = lyr.getGradBias();
					#ifdef _OPENMP
					#pragma omp parallel for schedule( static )
					#endif
				    for ( std::size_t i = 0; i < b.size(); ++i )
					    b.data()[i] -= gb.data()[i] * learning_rate;
			    }
		    }() ) );
	    },
	    m_layers );
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
Tensor2D_t *ConvolutionalBox<TensorN_t, Tensor2D_t, Layers...>::forward()
{
	std::size_t const batch = this->getInput()->rows();
	ensure_buffers( batch );

	m_fwd_slots[0].assign( this->getInput()->data(), this->getInput()->size() );

	std::apply( []( auto &...lyr ) { ( lyr.forward(), ... ); }, m_layers );

	TensorN_t const &last     = m_fwd_slots.back();
	std::size_t const flat    = last.size() / batch;
	Tensor2D_t *output        = this->getOutput();
	if ( output->rows() != batch || output->cols() != flat ) {
		*output = Tensor2D_t( batch, flat );
	}
	output->assign( last.data(), last.size() );

	return output;
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
template <std::size_t... Is>
void ConvolutionalBox<TensorN_t, Tensor2D_t, Layers...>::backward_reverse(
	std::index_sequence<Is...> )
{
	( std::get<sizeof...( Layers ) - 1 - Is>( m_layers ).backward(), ... );
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
Tensor2D_t *ConvolutionalBox<TensorN_t, Tensor2D_t, Layers...>::backward()
{
	TensorN_t &last_bwd = m_bwd_slots.back();
	if ( last_bwd.shape() != m_fwd_slots.back().shape() ) {
		last_bwd = TensorN_t( m_fwd_slots.back().shape() );
	}
	last_bwd.assign( this->getGradInput()->data(), this->getGradInput()->size() );

	backward_reverse( std::make_index_sequence<sizeof...( Layers )>{} );

	TensorN_t const &first_bwd = m_bwd_slots.front();
	std::size_t const flat_in  = first_bwd.size() / m_cached_batch_size;
	Tensor2D_t *grad_output    = this->getGradOutput();
	if ( grad_output->rows() != m_cached_batch_size || grad_output->cols() != flat_in ) {
		*grad_output = Tensor2D_t( m_cached_batch_size, flat_in );
	}
	grad_output->assign( first_bwd.data(), first_bwd.size() );

	return grad_output;
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
template <typename UnaryOp>
void ConvolutionalBox<TensorN_t, Tensor2D_t, Layers...>::forEachLayer( UnaryOp &&op )
{
	std::apply( [&op]( auto &...lyr ) { ( op( lyr ), ... ); }, m_layers );
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
template <typename Tuple, typename F>
void ConvolutionalBox<TensorN_t, Tensor2D_t, Layers...>::forEachLayerZip( Tuple &zip_tuple, F &&f )
{
	std::apply(
	    [&f, &zip_tuple]( auto &...lyr ) {
		    std::apply(
		        [&f, &lyr...]( auto &...z ) { (void)( ( f( lyr, z ), ... ), 0 ); },
		        zip_tuple );
	    },
	    m_layers );
}

namespace detail {

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
std::pair<std::size_t, std::size_t> next_slot_shape(
	ConvolutionalBox<TensorN_t, Tensor2D_t, Layers...> const &box,
	std::pair<std::size_t, std::size_t> const &in )
{
	return { in.first, box.outputCols() };
}

} // namespace detail

} // namespace neural

#endif // CONVOLUTIONAL_BOX_HPP
