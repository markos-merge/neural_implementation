#ifndef CONVOLUTIONAL_BOX_STATIC_HPP
#define CONVOLUTIONAL_BOX_STATIC_HPP

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include "batch_norm_layer.hpp"
#include "convolutional_layer.hpp"
#include "layer_base.hpp"
#include "dropout_layer.hpp"
#include "max_pool_layer.hpp"
#include "relu_layer.hpp"
#if NEURAL_CUDA_ENABLED
#include "cuda_tensor.hpp"
#include "cuda_tensor_n.hpp"
#include "neural_cuda_layer_sync.hpp"
#endif

namespace neural {

namespace detail {

template < typename TensorN_t >
std::array<std::size_t, 4> infer_shape_after( std::array<std::size_t, 4> s,
                                              ConvolutionalLayer<TensorN_t> const &L )
{
	std::size_t const k = L.kernelSize();
#if NEURAL_CUDNN_ENABLED
	// CudaTensor4::im2ColConvolution uses fixed padding 1,1 in cuDNN (stride 1) → H_out = H - k + 3.
	if constexpr ( is_cuda_tensor4_v<TensorN_t> ) {
		return { s[0], L.outChannels(), s[2] + 3 - k, s[3] + 3 - k };
	}
#endif
	std::size_t const pad = k / 2;
	return { s[0], L.outChannels(), s[2] - 2 * pad, s[3] - 2 * pad };
}

template < typename TensorN_t >
std::array<std::size_t, 4> infer_shape_after( std::array<std::size_t, 4> s,
                                              ReLULayer<TensorN_t> const & )
{
	return s;
}

template < typename TensorN_t >
std::array<std::size_t, 4> infer_shape_after( std::array<std::size_t, 4> s,
                                              DropoutLayer<TensorN_t> const & )
{
	return s;
}

template < typename TensorN_t >
std::array<std::size_t, 4> infer_shape_after( std::array<std::size_t, 4> s,
                                              BatchNormLayer<TensorN_t> const & )
{
	return s;
}

template < typename TensorN_t >
std::array<std::size_t, 4> infer_shape_after( std::array<std::size_t, 4> s,
                                              MaxPoolLayer<TensorN_t> const &L )
{
	std::size_t const oh = ( s[2] - L.poolSize() ) / L.stride() + 1;
	std::size_t const ow = ( s[3] - L.poolSize() ) / L.stride() + 1;
	return { s[0], s[1], oh, ow };
}

template < typename TensorN_t, typename... Layers >
std::size_t infer_flat_from_stack( std::size_t c, std::size_t h, std::size_t w,
                                   std::tuple<Layers...> const &layers )
{
	std::array<std::size_t, 4> s{ 1u, c, h, w };
	std::apply(
	    [&s]( auto const &...lyr ) {
		    ( ( s = infer_shape_after<TensorN_t>( s, lyr ) ), ... );
	    },
	    layers );
	return s[1] * s[2] * s[3];
}

} // namespace detail

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
class ConvolutionalBox_static : public LayerBase<Tensor2D_t>
{
	static_assert( TensorN_t::rank_v == 4, "ConvolutionalBox requires a rank-4 TensorN" );
	public:
		using tensor_tuple_t = std::tuple<Layers...>;

		using value_type = typename Tensor2D_t::value_type;
	public:

		/// \p channels, \p height, \p width describe the NCHW input; flattened output width is inferred from \p layers.
		ConvolutionalBox_static( std::size_t channels, std::size_t height, std::size_t width, Layers... layers );

		ConvolutionalBox_static( ConvolutionalBox_static const &other );
		ConvolutionalBox_static( ConvolutionalBox_static &&other ) noexcept;
		ConvolutionalBox_static &operator=( ConvolutionalBox_static const &other );
		ConvolutionalBox_static &operator=( ConvolutionalBox_static &&other ) noexcept;

		void initialize();

		Tensor2D_t *forward();
		Tensor2D_t *backward();

		std::size_t outputCols() const { return m_flat_output_cols; }

		template <typename UnaryOp>
		void forEachLayer( UnaryOp &&op );

		/// For train vs eval: forward to batch norm / dropout and any other layer that supports it.
		void setTraining( bool training );

		/// Zip each inner layer with the matching element of \p zip_tuple (same arity as inner \c Layers...).
		template <typename Tuple, typename F>
		void forEachLayerZip( Tuple &zip_tuple, F &&f );

	private:
		void wire_layers();
		void ensure_buffers( std::size_t batch_size );

		/// NCHW shape after the inner layer stack for batch size \p batch (same logic as \c m_flat_output_cols).
		std::array<std::size_t, 4> stacked_nchw_shape( std::size_t batch ) const;

		template <std::size_t... Is>
		void forward_inner( std::index_sequence<Is...> );

		template <std::size_t I>
		void forward_one_inner();

		template <std::size_t... Is>
		void backward_reverse( std::index_sequence<Is...> );

		template <std::size_t I>
		void backward_one_inner();

		std::size_t m_channels;
		std::size_t m_height;
		std::size_t m_width;

		std::tuple<Layers...> m_layers;
		std::size_t m_flat_output_cols;

		std::array<TensorN_t, sizeof...( Layers ) + 1> m_fwd_slots;
		std::array<TensorN_t, sizeof...( Layers ) + 1> m_bwd_slots;

		std::size_t m_cached_batch_size = 0;
};

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...>::ConvolutionalBox_static(
	std::size_t channels, std::size_t height, std::size_t width, Layers... layers )
	: m_channels( channels )
	, m_height( height )
	, m_width( width )
	, m_layers( std::forward<Layers>( layers )... )
	, m_flat_output_cols( detail::infer_flat_from_stack<TensorN_t>( channels, height, width, m_layers ) )
{
	wire_layers();
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...>::ConvolutionalBox_static( ConvolutionalBox_static const &other )
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
ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...>::ConvolutionalBox_static( ConvolutionalBox_static &&other ) noexcept
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
ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...> &
ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...>::operator=( ConvolutionalBox_static const &other )
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
ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...> &
ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...>::operator=( ConvolutionalBox_static &&other ) noexcept
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
std::array<std::size_t, 4>
ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...>::stacked_nchw_shape( std::size_t batch ) const
{
	std::array<std::size_t, 4> s{ batch, m_channels, m_height, m_width };
	std::apply(
	    [this, &s]( auto const &... lyr ) {
		    ( ( s = detail::infer_shape_after<TensorN_t>( s, lyr ) ), ... );
	    },
	    m_layers );
	return s;
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
void ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...>::wire_layers()
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

	if ( LayerBase<Tensor2D_t>::getInput() != nullptr ) {
		Tensor2D_t *const in = LayerBase<Tensor2D_t>::getInput();
		TensorN_t const in_view(
		    in->data(),
		    std::array<std::size_t, 4>{
		        static_cast<std::size_t>( in->rows() ), m_channels, m_height, m_width },
		    false );
		m_fwd_slots.front() = in_view;
	}

	if ( LayerBase<Tensor2D_t>::getOutput() != nullptr ) {
		Tensor2D_t *const out = LayerBase<Tensor2D_t>::getOutput();
		std::array<std::size_t, 4> const s =
		    stacked_nchw_shape( static_cast<std::size_t>( out->rows() ) );
		TensorN_t const out_view( out->data(), s, false );
		m_fwd_slots.back() = out_view;
	}

	if ( LayerBase<Tensor2D_t>::getGradInput() != nullptr ) {
		Tensor2D_t *const gin = LayerBase<Tensor2D_t>::getGradInput();
		std::array<std::size_t, 4> const s =
		    stacked_nchw_shape( static_cast<std::size_t>( gin->rows() ) );
		TensorN_t const gin_view( gin->data(), s, false );
		m_bwd_slots.back() = gin_view;
	}

	if ( LayerBase<Tensor2D_t>::getGradOutput() != nullptr ) {
		Tensor2D_t *const gout = LayerBase<Tensor2D_t>::getGradOutput();
		TensorN_t const gout_view(
		    gout->data(),
		    std::array<std::size_t, 4>{
		        static_cast<std::size_t>( gout->rows() ), m_channels, m_height, m_width },
		    false );
		m_bwd_slots.front() = gout_view;
	}
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
void ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...>::ensure_buffers( std::size_t batch_size )
{
	// if ( batch_size == m_cached_batch_size )
	// 	return;
	if ( LayerBase<Tensor2D_t>::getInput() != nullptr ) {
		Tensor2D_t *const in = LayerBase<Tensor2D_t>::getInput();
		TensorN_t const slot_view(
		    in->data(),
		    std::array<std::size_t, 4>{ batch_size, m_channels, m_height, m_width },
		    false );
		m_fwd_slots.front() = slot_view;
	}
	if ( LayerBase<Tensor2D_t>::getOutput() != nullptr ) {
		Tensor2D_t *const out = LayerBase<Tensor2D_t>::getOutput();
		std::array<std::size_t, 4> const s = stacked_nchw_shape( batch_size );
		TensorN_t const out_view( out->data(), s, false );
		m_fwd_slots.back() = out_view;
	}
	if ( LayerBase<Tensor2D_t>::getGradInput() != nullptr ) {
		Tensor2D_t *const gin = LayerBase<Tensor2D_t>::getGradInput();
		std::size_t const gb = static_cast<std::size_t>( gin->rows() );
		std::array<std::size_t, 4> const s = stacked_nchw_shape( gb );
		TensorN_t const gin_view( gin->data(), s, false );
		m_bwd_slots.back() = gin_view;
	}
	if ( LayerBase<Tensor2D_t>::getGradOutput() != nullptr ) {
		Tensor2D_t *const gout = LayerBase<Tensor2D_t>::getGradOutput();
		std::size_t const gb = static_cast<std::size_t>( gout->rows() );
		TensorN_t const gout_view(
		    gout->data(),
		    std::array<std::size_t, 4>{ gb, m_channels, m_height, m_width },
		    false );
		m_bwd_slots.front() = gout_view;
	}
	m_cached_batch_size = batch_size;
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
void ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...>::initialize()
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
Tensor2D_t *ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...>::forward()
{
	std::size_t const batch  = this->getInput()->rows();
	Tensor2D_t *const output = this->getOutput();
	std::size_t const flat   = m_flat_output_cols;
	if ( output->rows() != batch || output->cols() != flat ) {
		*output = Tensor2D_t( batch, flat );
	}
	ensure_buffers( batch );

	typename Tensor2D_t::value_type *const in_data = this->getInput()->data();
	if ( m_fwd_slots.front().data() != in_data
	     || m_fwd_slots.front().size() != this->getInput()->size() ) {
		m_fwd_slots[0].assign( in_data, this->getInput()->size() );
	}

	forward_inner( std::make_index_sequence<sizeof...( Layers )>{} );

	TensorN_t const &last = m_fwd_slots.back();
	if ( output->data() != last.data() || output->size() != last.size() ) {
		output->assign( last.data(), last.size() );
	}

#if NEURAL_CUDA_ENABLED
	if constexpr ( is_cuda_tensor_v<Tensor2D_t> || is_cuda_tensor4_v<TensorN_t> ) {
		cuda_layer_sync();
	}
#endif
	return output;
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
template <std::size_t I>
void ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...>::forward_one_inner()
{
	std::get<I>( m_layers ).forward();
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
template <std::size_t... Is>
void ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...>::forward_inner(
	std::index_sequence<Is...> )
{
	(void)( ( ( forward_one_inner<Is>(), 0 ), ... ) );
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
template <std::size_t I>
void ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...>::backward_one_inner()
{
	std::get<I>( m_layers ).backward();
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
template <std::size_t... Is>
void ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...>::backward_reverse(
	std::index_sequence<Is...> )
{
	(void)( ( ( backward_one_inner<sizeof...( Layers ) - 1u - Is>(), 0 ), ... ) );
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
Tensor2D_t *ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...>::backward()
{
	Tensor2D_t *const grad_output = this->getGradOutput();
	std::size_t const flat_in     = m_channels * m_height * m_width;
	if ( grad_output->rows() != m_cached_batch_size || grad_output->cols() != flat_in ) {
		*grad_output = Tensor2D_t( m_cached_batch_size, flat_in );
	}
	ensure_buffers( m_cached_batch_size );

	TensorN_t &last_bwd = m_bwd_slots.back();
	if ( last_bwd.shape() != m_fwd_slots.back().shape() ) {
		last_bwd.throw_if_non_owning_realloc(
		    "ConvolutionalBox::backward: cannot resize non-owning grad slot to match activations" );
		TensorN_t const resized( m_fwd_slots.back().shape() );
		last_bwd = resized;
	}
	if ( last_bwd.data() != this->getGradInput()->data()
	     || last_bwd.size() != this->getGradInput()->size() ) {
		last_bwd.assign( this->getGradInput()->data(), this->getGradInput()->size() );
	}

	backward_reverse( std::make_index_sequence<sizeof...( Layers )>{} );

	TensorN_t const &first_bwd = m_bwd_slots.front();
	if ( grad_output->data() != first_bwd.data()
	     || grad_output->size() != first_bwd.size() ) {
		grad_output->assign( first_bwd.data(), first_bwd.size() );
	}

#if NEURAL_CUDA_ENABLED
	if constexpr ( is_cuda_tensor_v<Tensor2D_t> || is_cuda_tensor4_v<TensorN_t> ) {
		cuda_layer_sync();
	}
#endif
	return grad_output;
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
template <typename UnaryOp>
void ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...>::forEachLayer( UnaryOp &&op )
{
	std::apply( [&op]( auto &...lyr ) { ( op( lyr ), ... ); }, m_layers );
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
void ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...>::setTraining( bool training )
{
	forEachLayer( [training]( auto &lyr ) {
		if constexpr ( requires { lyr.setTraining( training ); } ) {
			lyr.setTraining( training );
		}
	} );
}

template <typename TensorN_t, typename Tensor2D_t, typename... Layers>
template <typename Tuple, typename F>
void ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...>::forEachLayerZip( Tuple &zip_tuple, F &&f )
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
	ConvolutionalBox_static<TensorN_t, Tensor2D_t, Layers...> const &box,
	std::pair<std::size_t, std::size_t> const &in )
{
	return { in.first, box.outputCols() };
}

} // namespace detail

} // namespace neural

#endif // CONVOLUTIONAL_BOX_STATIC_HPP
