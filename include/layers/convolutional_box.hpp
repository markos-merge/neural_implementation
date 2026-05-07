#ifndef CONVOLUTIONAL_BOX_HPP
#define CONVOLUTIONAL_BOX_HPP

#include <array>
#include <cstddef>
#include <vector>
#include "conv_layer_variant.hpp"
#include "layer_base.hpp"

namespace neural {

// forward declaration required by detail::next_slot_shape below
template <typename TensorN_t, typename Tensor2D_t>
class ConvolutionalBox;

namespace detail {

/// Walk a runtime vector of ConvLayer variants and propagate the NCHW shape through each.
template <typename TensorN_t>
std::array<std::size_t, 4> infer_shape_after_variant( std::array<std::size_t, 4> s,
                                                       ConvLayer<TensorN_t> const &lyr );

template <typename TensorN_t>
std::size_t infer_flat_from_stack( std::size_t c, std::size_t h, std::size_t w,
                                   std::vector<ConvLayer<TensorN_t>> const &layers );

template <typename TensorN_t, typename Tensor2D_t>
std::pair<std::size_t, std::size_t> next_slot_shape(
    ConvolutionalBox<TensorN_t, Tensor2D_t> const &box,
    std::pair<std::size_t, std::size_t> const &in );

} // namespace detail

/// Dynamic ConvolutionalBox: inner layer stack stored as std::vector<ConvLayer<TensorN_t>>.
/// The layer sequence is fixed at construction time but not encoded in the type,
/// which makes it suitable for serialisation and runtime-built networks.
template <typename TensorN_t, typename Tensor2D_t>
class ConvolutionalBox : public LayerBase<Tensor2D_t>
{
	static_assert( TensorN_t::rank_v == 4, "ConvolutionalBox requires a rank-4 TensorN" );

	public:
		using value_type = typename Tensor2D_t::value_type;
		using layer_t    = ConvLayer<TensorN_t>;

		ConvolutionalBox();
		ConvolutionalBox( std::size_t channels, std::size_t height, std::size_t width );

		ConvolutionalBox( ConvolutionalBox const &other );
		ConvolutionalBox( ConvolutionalBox &&other ) noexcept;
		ConvolutionalBox &operator=( ConvolutionalBox const &other );
		ConvolutionalBox &operator=( ConvolutionalBox &&other ) noexcept;

		void initialize();
	
		void addLayer( ConvLayer<TensorN_t> &&layer );
		void clear();

		Tensor2D_t *forward();
		Tensor2D_t *backward();

		std::size_t outputCols() const { return m_flat_output_cols; }

		void forEachLayer( std::function<void(layer_t &)> &&op );

		/// Forward setTraining to BatchNormLayer / DropoutLayer inside the stack.
		void setTraining( bool training );
	private:
		void wire_layers();
		void ensure_buffers( std::size_t batch_size );

		std::array<std::size_t, 4> stacked_nchw_shape( std::size_t batch ) const;

		std::size_t m_channels;
		std::size_t m_height;
		std::size_t m_width;

		std::vector<layer_t>   m_layers;
		std::size_t            m_flat_output_cols;

		std::vector<TensorN_t> m_fwd_slots;   // size == m_layers.size() + 1
		std::vector<TensorN_t> m_bwd_slots;   // size == m_layers.size() + 1

		std::size_t m_cached_batch_size = 0;
};

//Implementation
template <typename TensorN_t, typename Tensor2D_t>
ConvolutionalBox<TensorN_t, Tensor2D_t>::ConvolutionalBox()
	: m_channels( 0 )
	, m_height( 0 )
	, m_width( 0 )
	, m_flat_output_cols( 0 )
{
	this->setLayerContainer( true );
}

template <typename TensorN_t, typename Tensor2D_t>
ConvolutionalBox<TensorN_t, Tensor2D_t>::ConvolutionalBox( std::size_t channels, std::size_t height, std::size_t width )
	: m_channels( channels )
	, m_height( height )
	, m_width( width )
	, m_flat_output_cols( 0 )
{
	this->setLayerContainer( true );
}

template <typename TensorN_t, typename Tensor2D_t>
ConvolutionalBox<TensorN_t, Tensor2D_t>::ConvolutionalBox( ConvolutionalBox const &other )
	: LayerBase<Tensor2D_t>( other )
	, m_channels( other.m_channels )
	, m_height( other.m_height )
	, m_width( other.m_width )
	, m_flat_output_cols( other.m_flat_output_cols )
	, m_layers( other.m_layers )
{
}

template <typename TensorN_t, typename Tensor2D_t>
ConvolutionalBox<TensorN_t, Tensor2D_t>::ConvolutionalBox( ConvolutionalBox &&other ) noexcept
	: LayerBase<Tensor2D_t>( std::move( other ) )
	, m_channels( other.m_channels )
	, m_height( other.m_height )
	, m_width( other.m_width )
	, m_flat_output_cols( other.m_flat_output_cols )
	, m_layers( std::move( other.m_layers ) )
{
}

template <typename TensorN_t, typename Tensor2D_t>
ConvolutionalBox<TensorN_t, Tensor2D_t> &ConvolutionalBox<TensorN_t, Tensor2D_t>::operator=( ConvolutionalBox const &other )
{
	LayerBase<Tensor2D_t>::operator=( other );
	m_channels = other.m_channels;
	m_height = other.m_height;
	m_width = other.m_width;
	m_flat_output_cols = other.m_flat_output_cols;
	m_layers = other.m_layers;

	return *this;
}

template <typename TensorN_t, typename Tensor2D_t>
ConvolutionalBox<TensorN_t, Tensor2D_t> &ConvolutionalBox<TensorN_t, Tensor2D_t>::operator=( ConvolutionalBox &&other ) noexcept
{
	LayerBase<Tensor2D_t>::operator=( std::move( other ) );
	m_channels = other.m_channels;
	m_height = other.m_height;
	m_width = other.m_width;
	m_flat_output_cols = other.m_flat_output_cols;
	m_layers = std::move( other.m_layers );
	
	return *this;
}


} // namespace neural

#endif // CONVOLUTIONAL_BOX_HPP
