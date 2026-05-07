#ifndef NEURAL_IMPL_TENSOR_N_HPP
#define NEURAL_IMPL_TENSOR_N_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <vector>
#include <iterator>
#include <numeric>
#include <array>
#include <Eigen/Dense>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace neural {

/// Validates \p v has length \p Rank and returns it as row-major extents.
template <std::size_t Rank>
inline std::array<std::size_t, Rank> nn_shape_vec_to_fixed( std::vector<std::size_t> const &v )
{
	if ( v.size() != Rank ) {
		throw std::invalid_argument( "tensor shape vector size must equal static rank" );
	}
	std::array<std::size_t, Rank> out{};
	std::copy( v.begin(), v.end(), out.begin() );
	return out;
}

template <typename T>
class Tensor;

// All images are 4D tensors (NCHW). 
// -N: number of images in the batch
// -C: number of channels
// -H: height
// -W: width
// So the shape of the std::array for the dims are {N, C, H, W}.
// array[0] = N, array[1] = C, array[2] = H, array[3] = W.
// The convolution operation is always applied without padding, so the shape of the output tensor
// is always smaller than the shape of the input tensor, depending on the kernel size

template <std::size_t rank, typename T = float>
class TensorN
{
	public:
		using value_type = T;
		static constexpr std::size_t rank_v = rank;
		template <std::size_t other_rank, typename type >
		using tensor_alias = TensorN<other_rank, type>;
	
		TensorN() noexcept;
		TensorN( std::array<std::size_t, rank> shape );
		TensorN( std::array< std::size_t, rank > shape, value_type value );
		template <std::random_access_iterator It>
		TensorN( std::array<std::size_t, rank> shape, It begin, It end );
		explicit TensorN( std::vector< value_type > const &data, std::array< std::size_t, rank > const &shape );
		/// Wrap \p buffer with given \p shape; \p own \c false = non-owning view (storage not freed here).
		/// If \p own is \c true, copies \p buffer into owned \c std::vector storage (caller keeps \p buffer).
		TensorN( value_type *buffer, std::array<std::size_t, rank> shape, bool own ) noexcept;
		/// Same as `TensorN(buffer, array, own)`; \p shape must have length \c rank.
		TensorN( value_type *buffer, std::vector<std::size_t> const &shape, bool own );
		
		TensorN( TensorN const &other );
		TensorN( TensorN &&other ) noexcept;
		TensorN &operator=( TensorN const &other );
		TensorN &operator=( TensorN &&other );
		~TensorN() = default;

		void throw_if_non_owning_realloc( char const *context ) const
		{
			if ( !m_own ) {
				throw std::invalid_argument( context );
			}
		}
		
		value_type *data() noexcept;
		value_type const *data() const noexcept;
		std::array<std::size_t, rank> shape() const noexcept;
		std::array<std::size_t, rank> strides() const noexcept;
		std::size_t size() const noexcept;
	
		value_type operator()( std::array< std::size_t, rank > const &indices ) const;
		
		void reshape( std::array< std::size_t, rank > const &new_shape );
		void swapAxes( std::array< std::size_t, rank > const &new_axes, TensorN< rank, T > &output );
	
		TensorN &operator*=( TensorN const &other );
		TensorN &operator*=( value_type scalar );
		/// \p out = \c *this ⊙ \p other; resizes \p out if needed (same as \c Tensor::elementwiseMultiply).
		void elementwiseMultiply( TensorN const &other, TensorN &out ) const;
		TensorN &cwiseGreaterInPlace( TensorN const &other, value_type scalar );
		TensorN &mulNSubstractInPlace( TensorN const &other, value_type scalar );

		value_type       &at( std::size_t n, std::size_t c, std::size_t h, std::size_t w ) requires ( rank == 4 );
		value_type const &at( std::size_t n, std::size_t c, std::size_t h, std::size_t w ) const requires ( rank == 4 );

		void addColorChannelInPlace( TensorN<1, T> const &bias ) requires ( rank == 4 );
		/// Bias with shape `{1, C, 1, 1}` matching \p m_shape[1] (NCHW).
		void addColorChannelInPlace( TensorN<4, T> const &bias ) requires ( rank == 4 );

		void addElementwise( std::array< std::size_t, rank > const &indices, value_type value );
		void assign( std::array< std::size_t, rank > const &indices, value_type value );
		void assign( std::vector< value_type > const &data );
		void assign( value_type const *src, std::size_t size );

		template< std::size_t other_rank >
		void assign( std::array< std::size_t, rank > const &from_slice, std::array< std::size_t, rank > const &to_slice, TensorN< other_rank, T > const &tensor );
	
		void reduceSumToDim( std::size_t axis, TensorN< 1, T > &output );
		/// Same sum as the rank-1 overload; result stored with shape `{1, K, 1, 1}` where \p K is the
		/// reduced extent along \p axis (only \p axis == 0 implemented).
		void reduceSumToDim( std::size_t axis, TensorN< 4, T > &output ) requires ( rank == 4 );
		

		template <typename Generator>
		void randomize( Generator &generator );
		/// Seeded uniform \f$[0, 1)\f$; deterministic given \p seed.
		void randomize( std::uint64_t seed );
		void randomizeHe( std::size_t fan_in, std::uint64_t seed );
		/// PyTorch \c nn.Conv* default; see \c Tensor::randomizePytorchDefault.
		void randomizePytorchDefault( std::size_t fan_in, std::uint64_t seed );
		// im2col convolution: input (*this), kernel, workspace, and output are all rank-4 (e.g. NCHW).
		void im2Col( std::array< std::size_t, 3 > const &kernel_shape, TensorN< 2, T > &output ) const requires ( rank == 4 );
		void multiply( TensorN< 2, T > const &tensor_2d, bool transpose_second, TensorN< 4, T > &output ) const requires ( rank == 4 );
		/// \p gemm_cnhw_layout reuses storage for the CNHW GEMM result; resized only when its shape
		/// differs from \code {C_out, N, H_out, W_out} \endcode.
		void im2ColConvolution( TensorN< 4, T > const &kernel, TensorN< 2, T > &im2col_tensor,
		                        TensorN< 4, T > &output, TensorN< 4, T > &gemm_cnhw_layout )
			requires ( rank == 4 );
		void col2Im( std::array< std::size_t, 4 > const &kernel_shape, std::array< std::size_t, 4 > original_shape, TensorN< 4, T > &output ) requires ( rank == 4 );
	protected:
		std::vector<value_type> m_data;
	private:
		void recomputeStrides() noexcept;
		std::size_t index( std::array< std::size_t, rank > const &shape, std::array< std::size_t, rank > const &indices ) const;
		std::size_t index( std::array<std::size_t, rank> const indices ) const;

		template < typename Op >
		bool loopIndices( std::array<std::size_t, rank> const &from_indices, std::array< std::size_t, rank > const &to_indices, Op op );

		/// Same visit order as \ref loopIndices, but splits axis 0 (batch) across OpenMP threads.
		/// When \c _OPENMP is off, behaves like \ref loopIndices. The callback must be safe to run
		/// concurrently for distinct batch indices (no conflicting writes). Early exit via returning
		/// \c true from \p op is only honored in the sequential fallback.
		template < typename Op >
		bool loopIndicesParallel( std::array<std::size_t, rank> const &from_indices, std::array< std::size_t, rank > const &to_indices, Op op );

	private:
		std::array<std::size_t, rank> m_shape{};
		std::array<std::size_t, rank> m_strides{};
		value_type *m_view = nullptr;
		bool m_own = true;
};

template < std::size_t rank, typename T >
TensorN<rank, T>::TensorN() noexcept
{
	recomputeStrides();
}

template < std::size_t rank, typename T >
TensorN<rank, T>::TensorN( std::array<std::size_t, rank> shape )
	: m_shape( shape )
{
	std::size_t const size = std::accumulate( shape.begin(), shape.end(), 1, std::multiplies<>() );
	m_data.resize( size, 0. );
	m_view = nullptr;
	m_own = true;
	recomputeStrides();
}

template < std::size_t rank, typename T >
TensorN<rank, T>::TensorN( std::array<std::size_t, rank> shape, value_type value )
	: m_shape( shape )
{
	std::size_t const size = std::accumulate( shape.begin(), shape.end(), 1, std::multiplies<>() );
	m_data.resize( size, value );
	m_view = nullptr;
	m_own = true;
	recomputeStrides();
}

template < std::size_t rank, typename T >
template <std::random_access_iterator It>
TensorN<rank, T>::TensorN( std::array<std::size_t, rank> shape, It begin, It end )
	: m_shape( shape )
{
	std::size_t const size = std::accumulate( shape.begin(), shape.end(), 1, std::multiplies<>() );
	if( std::distance( begin, end ) != size )
	{
		throw std::invalid_argument( "iterator range size does not match shape" );
	}
	m_data.reserve( size );
	std::copy( begin, end, std::back_inserter( m_data ) );
	m_view = nullptr;
	m_own = true;
	recomputeStrides();
}

template < std::size_t rank, typename T >
TensorN<rank, T>::TensorN( std::vector< value_type > const &data, std::array<std::size_t, rank> const &shape )
	: m_shape( shape )
{
	std::size_t const size = std::accumulate( shape.begin(), shape.end(), 1, std::multiplies<>() );
	if( data.size() != size )
	{
		throw std::invalid_argument( "data size does not match shape" );
	}

	m_data = data;
	m_view = nullptr;
	m_own = true;
	recomputeStrides();
}

template < std::size_t rank, typename T >
TensorN<rank, T>::TensorN( value_type *buffer, std::array<std::size_t, rank> shape, bool own ) noexcept
	: m_shape( shape )
{
	std::size_t const n = std::accumulate( shape.begin(), shape.end(), 1, std::multiplies<>() );
	if ( own ) {
		m_own = true;
		m_view = nullptr;
		m_data.resize( n );
		if ( n > 0U && buffer != nullptr ) {
			std::copy( buffer, buffer + n, m_data.begin() );
		}
	} else {
		m_own = false;
		m_view = buffer;
		m_data.clear();
	}
	recomputeStrides();
}

template < std::size_t rank, typename T >
TensorN<rank, T>::TensorN( value_type *buffer, std::vector<std::size_t> const &shape, bool own )
	: TensorN( buffer, nn_shape_vec_to_fixed<rank>( shape ), own )
{
}

template < std::size_t rank, typename T >
TensorN<rank, T>::TensorN( TensorN const &other )
	: m_shape( other.m_shape )
	, m_strides( other.m_strides )
	, m_view( nullptr )
	, m_own( true )
{
	m_data.assign( other.data(), other.data() + other.size() );
}

template < std::size_t rank, typename T >
TensorN<rank, T>::TensorN( TensorN &&other ) noexcept
	: m_shape( other.m_shape )
	, m_strides( other.m_strides )
	, m_own( other.m_own )
	, m_view( other.m_view )
{
	if ( other.m_own ) {
		m_data = std::move( other.m_data );
		m_view = nullptr;
	} else {
		m_data.clear();
	}
	other.m_view = nullptr;
	other.m_own = true;
	other.m_shape = {};
	other.m_data.clear();
	other.recomputeStrides();
}

template < std::size_t rank, typename T >
TensorN<rank, T> &TensorN<rank, T>::operator=( TensorN const &other )
{
	if ( m_own ) {
		if ( size() != other.size() ) {
			m_data.resize( other.size() );
		}
		std::copy( other.data(), other.data() + other.size(), m_data.begin() );
		m_shape = other.m_shape;
		m_strides = other.m_strides;
		m_view = nullptr;
		m_own = true;
		return *this;
	}
	if ( m_shape != other.m_shape ) {
		throw std::invalid_argument(
		    "TensorN::operator=(TensorN const &): non-owning tensor requires matching shape" );
	}
	std::copy( other.data(), other.data() + other.size(), data() );
	m_strides = other.m_strides;
	return *this;
}

template < std::size_t rank, typename T >
TensorN<rank, T> &TensorN<rank, T>::operator=( TensorN &&other )
{
	if ( this == &other ) {
		return *this;
	}
	throw_if_non_owning_realloc(
	    "TensorN::operator=(TensorN&&): cannot move-assign into a non-owning tensor" );
	m_shape = other.m_shape;
	m_strides = other.m_strides;
	m_own = other.m_own;
	m_view = other.m_view;
	if ( other.m_own ) {
		m_data = std::move( other.m_data );
		m_view = nullptr;
	} else {
		m_data.clear();
	}
	other.m_view = nullptr;
	other.m_own = true;
	other.m_shape = {};
	other.m_data.clear();
	other.recomputeStrides();
	return *this;
}

template < std::size_t rank, typename T >
std::array<std::size_t, rank> TensorN<rank, T>::shape() const noexcept
{
	return m_shape;
}

template < std::size_t rank, typename T >
std::array<std::size_t, rank> TensorN<rank, T>::strides() const noexcept
{
	return m_strides;
}

template < std::size_t rank, typename T >
T *TensorN<rank, T>::data() noexcept
{
	return m_own ? m_data.data() : m_view;
}

template < std::size_t rank, typename T >
T const *TensorN<rank, T>::data() const noexcept
{
	return m_own ? m_data.data() : m_view;
}

template < std::size_t rank, typename T >
std::size_t TensorN<rank, T>::size() const noexcept
{
	return std::accumulate( m_shape.begin(), m_shape.end(), 1, std::multiplies<>() );
}

template < std::size_t rank, typename T >
void TensorN<rank, T>::recomputeStrides() noexcept
{
	if constexpr ( rank > 0 ) {
		m_strides[rank - 1] = 1;
		for ( std::size_t i = rank - 1; i > 0; --i ) {
			m_strides[i - 1] = m_strides[i] * m_shape[i];
		}
	}
}

template < std::size_t rank, typename T >
T TensorN<rank, T>::operator()( std::array<std::size_t, rank> const &indices ) const
{
	return data()[index( indices )];
}

template< std::size_t rank, typename T >
std::size_t TensorN<rank, T>::index( std::array< std::size_t, rank > const &shape, std::array< std::size_t, rank > const &indices ) const
{
	std::size_t idx = 0;
	std::size_t stride = 1;
	for ( std::size_t i = rank; i-- > 0; ) {
		idx += indices[i] * stride;
		stride *= shape[i];
	}
	return idx;
}

template < std::size_t rank, typename T >
std::size_t TensorN<rank, T>::index( std::array<std::size_t, rank> const indices ) const
{
	std::size_t idx = 0;
	for ( std::size_t i = 0; i < rank; ++i ) {
		idx += indices[i] * m_strides[i];
	}
	return idx;
}

template < std::size_t rank, typename T >
void TensorN<rank, T>::reshape( std::array< std::size_t, rank > const &new_shape )
{
	if( std::accumulate( new_shape.begin(), new_shape.end(), 1, std::multiplies<>() ) != size() )
	{
		throw std::invalid_argument( "new shape size does not match tensor size" );
	}
	m_shape = new_shape;
	recomputeStrides();
}

template < std::size_t rank, typename T >
void TensorN<rank, T>::swapAxes( std::array< std::size_t, rank > const &new_axes, TensorN< rank, T > &output )
{
	if( new_axes.size() != rank ) {
		throw std::invalid_argument( "new axes size does not match rank" );
	}

	std::array< std::size_t, rank > new_shape;
	for( std::size_t i = 0; i < rank; ++i ) {
		new_shape[i] = m_shape[new_axes[i]];
	}

	if( output.shape() != new_shape ) {
		output.throw_if_non_owning_realloc( "TensorN::swapAxes: cannot reallocate non-owning output" );
		output = TensorN< rank, T >( new_shape );
	}
	
	T *out_data = output.data();
	std::array< std::size_t, rank > &strides = output.m_strides;

	loopIndicesParallel( {}, m_shape, [this, &new_axes, &strides, &out_data]( std::array<std::size_t, rank> const &indices ){
		std::size_t idx = 0;
		std::size_t from_idx = 0;
			for( std::size_t i = 0; i < rank; ++i ) {
				idx += indices[new_axes[i]] * strides[i];
				from_idx += indices[i] * m_strides[i];
			}

			out_data[idx] = data()[from_idx];

			return false;
	} );

}

template < std::size_t rank, typename T >
void TensorN<rank, T>::assign( std::array< std::size_t, rank > const &indices, value_type value )
{
	data()[index( indices )] = value;
}

template < std::size_t rank, typename T >
void TensorN<rank, T>::assign( std::vector< value_type > const &data )
{
	if( data.size() != size() )
	{
		throw std::invalid_argument( "data size does not match shape" );
	}
	if ( m_own ) {
		m_data = data;
		m_view = nullptr;
	} else {
		std::copy( data.begin(), data.end(), this->data() );
	}
}

template < std::size_t rank, typename T >
void TensorN<rank, T>::assign( value_type const *src, std::size_t size )
{
	if( size != this->size() )
	{
		throw std::invalid_argument( "TensorN::assign(): size does not match tensor size" );
	}
	std::copy( src, src + size, this->data() );
}

namespace detail {

template < std::size_t Rank >
std::array< std::size_t, Rank > rowMajorStrides( std::array<std::size_t, Rank> const &shape )
{
	std::array<std::size_t, Rank> strides{};
	if constexpr ( Rank > 0 ) {
		strides[Rank - 1] = 1;
		for ( std::size_t i = Rank - 1; i > 0; --i ) {
			strides[i - 1] = strides[i] * shape[i];
		}
	}
	return strides;
}

template < std::size_t Rank >
std::array< std::size_t, Rank > linearToRowMajorIndices( std::size_t linear,
                                                         std::array< std::size_t, Rank > const &strides )
{
	std::array<std::size_t, Rank> out{};
	std::size_t remaining = linear;
	for ( std::size_t d = 0; d < Rank; ++d ) {
		std::size_t const s = strides[d];
		out[d] = s ? remaining / s : 0;
		remaining %= s;
	}
	return out;
}

// Row-major visit order (last index varies fastest), matching TensorN::index().
template <typename Op, std::size_t rank, typename T, std::size_t dim_index>
bool loopIndices( Op &&op, TensorN<rank, T> &tensor,
                  std::array<std::size_t, rank> const &from_indices,
                  std::array<std::size_t, rank> const &to_indices,
                  std::array<std::size_t, rank> &cur_index )
{
	bool ans = false;
	for ( std::size_t i = from_indices[dim_index]; i < to_indices[dim_index];
	      ++i ) {
		cur_index[dim_index] = i;
		if constexpr ( dim_index + 1 < rank ) {
			if ( ( ans = detail::loopIndices<Op, rank, T, dim_index + 1>(
			           std::forward<Op>( op ), tensor, from_indices, to_indices,
			           cur_index ) ) ) {
				break;
			}
		} else {
			if ( ( ans = op( cur_index ) ) ) {
				break;
			}
		}
	}
	return ans;
}

template < typename Op, std::size_t rank, typename T >
bool loopIndicesParallelBatch( Op op, TensorN<rank, T> &tensor,
                               std::array<std::size_t, rank> const &from_indices,
                               std::array<std::size_t, rank> const &to_indices )
{
	std::size_t const b0 = from_indices[0];
	std::size_t const e0 = to_indices[0];
	if ( b0 >= e0 ) {
		return false;
	}
#ifdef _OPENMP
	std::ptrdiff_t const n0 = static_cast<std::ptrdiff_t>( e0 - b0 );
	#pragma omp parallel for schedule( static ) firstprivate( op )
	for ( std::ptrdiff_t ii = 0; ii < n0; ++ii ) {
		std::size_t const i0 = b0 + static_cast<std::size_t>( ii );
		std::array<std::size_t, rank> cur_index{};
		cur_index[0] = i0;
		if constexpr ( rank == 1 ) {
			(void)op( cur_index );
		} else {
			(void)loopIndices<Op &, rank, T, 1>( op, tensor, from_indices, to_indices, cur_index );
		}
	}
	(void)tensor;
	return false;
#else
	auto cur_index = std::array<std::size_t, rank>{};
	return loopIndices<Op, rank, T, 0>( std::move( op ), tensor, from_indices, to_indices, cur_index );
#endif
}

} // namespace detail

template < std::size_t rank, typename T >
template < typename Op >
bool TensorN<rank, T>::loopIndices( std::array<std::size_t, rank> const &from_indices, std::array< std::size_t, rank > const &to_indices, Op op )
{
	auto cur_index = std::array<std::size_t, rank>{};
	return detail::loopIndices< Op, rank, T, 0 >( std::forward<Op>( op ), *this, from_indices, to_indices, cur_index );
}

template < std::size_t rank, typename T >
template < typename Op >
bool TensorN<rank, T>::loopIndicesParallel( std::array<std::size_t, rank> const &from_indices, std::array<std::size_t, rank> const &to_indices, Op op )
{
	return detail::loopIndicesParallelBatch( std::move( op ), *this, from_indices, to_indices );
}

template < std::size_t rank, typename T >
template < std::size_t other_rank >
void TensorN<rank, T>::assign( std::array< std::size_t, rank > const &from_slice, std::array< std::size_t, rank > const &to_slice, TensorN< other_rank, T > const &tensor )
{
	static_assert( rank >= other_rank, "rank must be greater than or equal to other_rank" );
	
	for( std::size_t i = 0; i < rank; ++i ) {
		if( from_slice[i] >= to_slice[i] ) {
			throw std::invalid_argument( "from_slice must be less than to_slice" );
		}
	}

	std::array< std::size_t, rank > diff;
	for( std::size_t i = 0; i < rank; ++i ) {
		diff[i] = to_slice[i] - from_slice[i];
	}
	
	std::size_t region_size = std::accumulate( diff.begin(), diff.end(), 1, std::multiplies<>() );
	if( region_size != tensor.size() ) {
		throw std::invalid_argument( "region size does not match tensor size" );
	}

	std::array<std::size_t, other_rank> const tensor_strides = tensor.strides();
	std::size_t linear = 0;
	loopIndices( from_slice, to_slice, [&linear, this, &tensor, &tensor_strides]( std::array<std::size_t, rank> const indices ){
		std::array<std::size_t, other_rank> const other_indices =
		    detail::linearToRowMajorIndices<other_rank>( linear++, tensor_strides );
		this->assign( indices, tensor( other_indices ) );

		return false;
	} );
}

template <std::size_t rank, typename T >
void TensorN<rank, T>::im2Col( std::array< std::size_t, 3 > const &kernel_shape, TensorN< 2, T > &output ) const
	requires ( rank == 4 )
{
	std::size_t const padding_i = kernel_shape[1] / 2;
	std::size_t const padding_j = kernel_shape[2] / 2;
	std::array< std::size_t, 2 > output_shape{ kernel_shape[0] * kernel_shape[1] * kernel_shape[2], 
		m_shape.front()*( m_shape[2] - 2*padding_i )*( m_shape[3] - 2*padding_j ) };
	if( output.shape() != output_shape ) {
		output.throw_if_non_owning_realloc( "TensorN::im2Col: cannot reallocate non-owning output" );
		output = TensorN< 2, T >( output_shape );
	}
	
	T *output_data = output.data();
	
	std::size_t const batches_stride = m_shape[1]*m_shape[2]*m_shape[3];
	std::size_t const channels_stride = m_shape[2]*m_shape[3];
	std::size_t const rows_stride = m_shape[3];
	
	std::size_t const output_col_stride = output.shape()[1];
	std::size_t const Kh              = kernel_shape[1];
	std::size_t const Kw              = kernel_shape[2];
	std::size_t const patch_size      = Kh * Kw;
	std::size_t const num_rows        = kernel_shape[0] * patch_size;
	std::size_t const H_out           = m_shape[2] - 2 * padding_i;
	std::size_t const W_out           = m_shape[3] - 2 * padding_j;

#ifdef _OPENMP
#pragma omp parallel for schedule( static )
#endif
	for ( std::size_t b = 0; b < m_shape[0]; ++b ) {
		for ( std::size_t cur_row = 0; cur_row < num_rows; ++cur_row ) {
			std::size_t const kernel_channel = cur_row / patch_size;
			std::size_t const rem = cur_row % patch_size;
			std::size_t const kernel_row = rem / Kw;
			std::size_t const kernel_col = rem % Kw;

			for ( std::size_t i = padding_i; i + padding_i < m_shape[2]; ++i ) {
				for ( std::size_t j = padding_j; j + padding_j < m_shape[3]; ++j ) {
					std::size_t const output_col =
					    b * H_out * W_out + ( i - padding_i ) * W_out + ( j - padding_j );
					std::size_t const cur_im_index =
					    b * batches_stride + kernel_channel * channels_stride +
					    ( i - padding_i + kernel_row ) * rows_stride + j - padding_j +
					    kernel_col;
					output_data[cur_row * output_col_stride + output_col] =
					    data()[cur_im_index];
				}
			}
		}
	}
}

template <std::size_t rank, typename T >
void TensorN<rank, T>::multiply( TensorN< 2, T > const &tensor_2d, bool transpose_second, TensorN< 4, T > &output ) const requires ( rank == 4 )
{
	Eigen::Map< const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > > kernel_matrix(
	    this->data(), this->shape()[0], this->shape()[1] * this->shape()[2] * this->shape()[3] );
	Eigen::Map< const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > > aux_tensor_matrix( tensor_2d.data(), tensor_2d.shape()[0], tensor_2d.shape()[1] );
	// Rows of the GEMM result match the left factor (this) row count — same as forward into (N,C,H,W)
	// and backward grad_W into (C_out,C_in,k,k) where C_out is shape[0].
	std::size_t const output_rows = this->shape()[0];
	if( output.size() % output_rows != 0 ) {
		throw std::invalid_argument( "multiply: output size is not divisible by first operand leading dimension" );
	}
	std::size_t const output_cols = output.size() / output_rows;
	Eigen::Map<Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> output_matrix( output.data(), output_rows, output_cols );

	if( transpose_second ) {
		output_matrix.noalias() = kernel_matrix * aux_tensor_matrix.transpose();
	} else {
		output_matrix.noalias() = kernel_matrix * aux_tensor_matrix;
	}
	
}

template <std::size_t rank, typename T >
void TensorN<rank, T>::im2ColConvolution(
	TensorN< 4, T > const &kernel,
	TensorN< 2, T > &im2col_tensor,
	TensorN< 4, T > &output,
	TensorN< 4, T > &gemm_cnhw_layout )
	requires ( rank == 4 )
{

	std::array< std::size_t, 3 > const kernel_shape{ kernel.shape()[1], kernel.shape()[2], kernel.shape()[3] };
	im2Col( kernel_shape, im2col_tensor );
	
	std::size_t const padding_i = kernel.shape()[2] / 2;
	std::size_t const padding_j = kernel.shape()[3] / 2;
	std::size_t const H_out = m_shape[2] - 2 * padding_i;
	std::size_t const W_out = m_shape[3] - 2 * padding_j;
	// GEMM fills row-major (C_out × N·H·W) = flat layout (C_out, N, H, W), not (N, C_out, H, W).
	std::array< std::size_t, 4 > const cnhw_shape{ kernel.shape()[0], m_shape[0], H_out, W_out };
	if( gemm_cnhw_layout.shape() != cnhw_shape ) {
		gemm_cnhw_layout.throw_if_non_owning_realloc(
		    "TensorN::im2ColConvolution: cannot reallocate non-owning gemm_cnhw_layout" );
		gemm_cnhw_layout = TensorN< 4, T >( cnhw_shape );
	}
	kernel.multiply( im2col_tensor, false, gemm_cnhw_layout );
	gemm_cnhw_layout.swapAxes( { 1, 0, 2, 3 }, output );
}

/// NCHW batch-norm forward.
template <typename T>
void batch_norm_forward_nchw( TensorN<4, T> const &input, TensorN<4, T> const &gamma,
                              TensorN<4, T> const &beta, TensorN<4, T> &running_mean,
                              TensorN<4, T> &running_var, TensorN<4, T> &last_running_mean,
                              TensorN<4, T> &last_running_var, T eps, T momentum, bool training,
                              bool is_first_forward, TensorN<4, T> &output )
{
	std::array<std::size_t, 4> const in_shape = input.shape();
	std::size_t const C = in_shape[1];
	auto const is_channel_param_shape = [C]( std::array<std::size_t, 4> const &shape ) {
		return shape[0] == 1 && shape[2] == 1 && shape[3] == 1 && shape[1] == C;
	};

	if ( !is_channel_param_shape( gamma.shape() ) || !is_channel_param_shape( beta.shape() ) ||
	     !is_channel_param_shape( running_mean.shape() ) ||
	     !is_channel_param_shape( running_var.shape() ) ||
	     !is_channel_param_shape( last_running_mean.shape() ) ||
	     !is_channel_param_shape( last_running_var.shape() ) ) {
		throw std::invalid_argument(
		    "batchNormForward: expected {1, C, 1, 1} for gamma/beta/running tensors with C == input channels" );
	}
	if ( output.shape() != in_shape ) {
		throw std::invalid_argument( "batchNormForward: output shape must match input shape" );
	}

	std::size_t const N = in_shape[0];
	std::size_t const H = in_shape[2];
	std::size_t const W = in_shape[3];
	std::size_t const M = N * H * W;
	std::array<std::size_t, 4> const in_strides = input.strides();

	T *const running_mean_data = running_mean.data();
	T *const running_var_data = running_var.data();
	T *const last_running_mean_data = last_running_mean.data();
	T *const last_running_var_data = last_running_var.data();

	if ( training || is_first_forward ) {
#ifdef _OPENMP
#pragma omp parallel for schedule( static )
#endif
		for ( std::size_t c = 0; c < C; ++c ) {
			T channel_sum = static_cast<T>( 0 );
			for ( std::size_t n = 0; n < N; ++n ) {
				for ( std::size_t h = 0; h < H; ++h ) {
					for ( std::size_t w = 0; w < W; ++w ) {
						channel_sum += input.at( n, c, h, w );
					}
				}
			}
			running_mean_data[c] = channel_sum / static_cast<T>( M );
		}

#ifdef _OPENMP
#pragma omp parallel for schedule( static )
#endif
		for ( std::size_t c = 0; c < C; ++c ) {
			T var_sum = static_cast<T>( 0 );
			T const mean_c = running_mean_data[c];
			for ( std::size_t n = 0; n < N; ++n ) {
				for ( std::size_t h = 0; h < H; ++h ) {
					for ( std::size_t w = 0; w < W; ++w ) {
						T const centered = input.at( n, c, h, w ) - mean_c;
						var_sum += centered * centered;
					}
				}
			}
			running_var_data[c] = var_sum / static_cast<T>( M );
		}
	} else {
		std::copy( last_running_mean_data, last_running_mean_data + C, running_mean_data );
		std::copy( last_running_var_data, last_running_var_data + C, running_var_data );
	}

	T const *const gamma_data = gamma.data();
	T const *const beta_data = beta.data();
	T const *const input_data = input.data();
	T *const output_data = output.data();

#ifdef _OPENMP
#pragma omp parallel for collapse( 2 ) schedule( static )
#endif
	for ( std::size_t n = 0; n < N; ++n ) {
		for ( std::size_t c = 0; c < C; ++c ) {
			T const mean_c = running_mean_data[c];
			T const inv_std = static_cast<T>( 1 ) / std::sqrt( running_var_data[c] + eps );
			T const gamma_c = gamma_data[c];
			T const beta_c = beta_data[c];
			for ( std::size_t h = 0; h < H; ++h ) {
				for ( std::size_t w = 0; w < W; ++w ) {
					std::size_t const idx = n * in_strides[0] + c * in_strides[1] +
					                        h * in_strides[2] + w;
					output_data[idx] = gamma_c * ( input_data[idx] - mean_c ) * inv_std + beta_c;
				}
			}
		}
	}

	// PyTorch / cuDNN convention: momentum is weight on the current batch estimate:
	//   running ← (1 − momentum) * running + momentum * batch
	if ( training ) {
#ifdef _OPENMP
#pragma omp parallel for schedule( static )
#endif
		for ( std::size_t c = 0; c < C; ++c ) {
			last_running_mean_data[c] = last_running_mean_data[c] * ( static_cast<T>( 1 ) - momentum ) +
			                            running_mean_data[c] * momentum;
			last_running_var_data[c] = last_running_var_data[c] * ( static_cast<T>( 1 ) - momentum ) +
			                           running_var_data[c] * momentum;
		}
	}
}

/// NCHW batch-norm backward.
/// \p grad_wrt_output is \f$\partial L/\partial y\f$ and \p grad_wrt_input is \f$\partial L/\partial x\f$.
template <typename T>
void batch_norm_backward_nchw( TensorN<4, T> const &grad_wrt_output, TensorN<4, T> const &input,
                               TensorN<4, T> const &gamma, TensorN<4, T> const &running_mean,
                               TensorN<4, T> const &running_var, T eps,
                               TensorN<4, T> &grad_gamma, TensorN<4, T> &grad_beta,
                               TensorN<4, T> &grad_wrt_input )
{
	std::array<std::size_t, 4> const in_shape = input.shape();
	std::size_t const C = in_shape[1];
	auto const is_channel_param_shape = [C]( std::array<std::size_t, 4> const &shape ) {
		return shape[0] == 1 && shape[2] == 1 && shape[3] == 1 && shape[1] == C;
	};

	if ( grad_wrt_output.shape() != in_shape ) {
		throw std::invalid_argument( "batch_norm_backward_nchw: grad_wrt_output shape must match input shape" );
	}
	if ( grad_wrt_input.shape() != in_shape ) {
		throw std::invalid_argument( "batch_norm_backward_nchw: grad_wrt_input shape must match input shape" );
	}
	if ( !is_channel_param_shape( gamma.shape() ) ||
	     !is_channel_param_shape( running_mean.shape() ) ||
	     !is_channel_param_shape( running_var.shape() ) ||
	     !is_channel_param_shape( grad_gamma.shape() ) ||
	     !is_channel_param_shape( grad_beta.shape() ) ) {
		throw std::invalid_argument(
		    "batch_norm_backward_nchw: expected {1, C, 1, 1} for gamma/running/grad tensors with C == input channels" );
	}

	std::size_t const N = in_shape[0];
	std::size_t const H = in_shape[2];
	std::size_t const W = in_shape[3];
	std::size_t const HW = H * W;
	std::size_t const M = N * HW;

	std::array<std::size_t, 4> const dy_strides = grad_wrt_output.strides();
	std::array<std::size_t, 4> const dx_strides = grad_wrt_input.strides();
	T const *const dy_data = grad_wrt_output.data();
	T const *const input_data = input.data();
	T *const dx_data = grad_wrt_input.data();
	T const *const gamma_data = gamma.data();
	T const *const mean_data = running_mean.data();
	T const *const var_data = running_var.data();
	T *const dgamma_data = grad_gamma.data();
	T *const dbeta_data = grad_beta.data();

#ifdef _OPENMP
#pragma omp parallel for schedule( static )
#endif
	for ( std::size_t c = 0; c < C; ++c ) {
		T beta_grad = static_cast<T>( 0 );
		T gamma_grad = static_cast<T>( 0 );
		T const inv_std = static_cast<T>( 1 ) / std::sqrt( var_data[c] + eps );
		T const mean_c = mean_data[c];

		T sum_dy = static_cast<T>( 0 );
		T sum_dy_xhat = static_cast<T>( 0 );
		for ( std::size_t n = 0; n < N; ++n ) {
			std::size_t const base = n * dy_strides[0] + c * dy_strides[1];
			for ( std::size_t i = 0; i < HW; ++i ) {
				std::size_t const idx = base + i;
				T const xhat = ( input_data[idx] - mean_c ) * inv_std;
				T const dy = dy_data[idx];
				beta_grad += dy;
				gamma_grad += dy * xhat;
				sum_dy += dy;
				sum_dy_xhat += dy * xhat;
			}
		}
		dbeta_data[c] = beta_grad;
		dgamma_data[c] = gamma_grad;

		T const scale = gamma_data[c] * inv_std / static_cast<T>( M );
		for ( std::size_t n = 0; n < N; ++n ) {
			std::size_t const base_dy = n * dy_strides[0] + c * dy_strides[1];
			std::size_t const base_dx = n * dx_strides[0] + c * dx_strides[1];
			for ( std::size_t i = 0; i < HW; ++i ) {
				std::size_t const idx_dy = base_dy + i;
				std::size_t const idx_dx = base_dx + i;
				T const xhat = ( input_data[idx_dy] - mean_c ) * inv_std;
				dx_data[idx_dx] = scale * ( static_cast<T>( M ) * dy_data[idx_dy] - sum_dy -
				                            xhat * sum_dy_xhat );
			}
		}
	}
}

template< std::size_t rank, typename T >
template <typename Generator>
void TensorN< rank, T >::randomize( Generator &generator )
{
	std::generate( data(), data() + size(), generator );
}

template< std::size_t rank, typename T >
void TensorN< rank, T >::randomize( std::uint64_t seed )
{
	std::mt19937_64 gen( seed );
	std::uniform_real_distribution<double> dis( 0.0, 1.0 );
	auto generator = [&]() { return static_cast<T>( dis( gen ) ); };
	randomize( generator );
}

template< std::size_t rank, typename T >
void TensorN< rank, T >::randomizeHe( std::size_t fan_in, std::uint64_t seed )
{
	double const scale = std::sqrt( 2.0 / static_cast<double>( fan_in ) );
	std::mt19937_64 gen( seed );
	std::uniform_real_distribution<double> dis( -scale, scale );
	auto generator = [&]() { return static_cast<T>( dis( gen ) ); };

	std::generate( data(), data() + size(), generator );
}

template< std::size_t rank, typename T >
void TensorN< rank, T >::randomizePytorchDefault( std::size_t fan_in, std::uint64_t seed )
{
	if ( fan_in == 0 ) {
		return;
	}
	double const inv_sqrt = 1.0 / std::sqrt( static_cast<double>( fan_in ) );
	std::mt19937_64 gen( seed );
	std::uniform_real_distribution<double> dis( -inv_sqrt, inv_sqrt );
	auto generator = [&]() { return static_cast<T>( dis( gen ) ); };
	std::generate( data(), data() + size(), generator );
}

//different from reduceSum, this function reduces the tensor to a single dimension
template< std::size_t rank, typename T >
void TensorN< rank, T >::reduceSumToDim( std::size_t axis, TensorN< 1, T > &output )
{
	std::size_t const output_size = m_shape[axis];
	if( output.shape()[0] != output_size ) {
		output.throw_if_non_owning_realloc( "TensorN::reduceSumToDim: cannot reallocate non-owning output" );
		output = TensorN< 1, T >( { output_size } );
	}
	
	if( axis == 0 ) {
		T *out_ptr = output.data();
		std::size_t const stride = m_strides[0];
		T const *const self_base = this->data();
		
		#if _OPENMP
		#pragma omp parallel for schedule( static )
		#endif
		for( std::size_t i = 0; i < output.size(); ++i ) {
			out_ptr[i] = std::accumulate( self_base + i * stride, self_base + ( i + 1 ) * stride, 0.0 );
		}
	} else {
		//todo: implement this
	}
	
}

template< std::size_t rank, typename T >
void TensorN< rank, T >::reduceSumToDim( std::size_t axis, TensorN< 4, T > &output ) requires ( rank == 4 )
{
	std::size_t const output_size = m_shape[axis];
	std::array< std::size_t, 4 > const out_shape{ 1, output_size, 1, 1 };
	if( output.shape() != out_shape ) {
		output.throw_if_non_owning_realloc( "TensorN::reduceSumToDim: cannot reallocate non-owning output" );
		output = TensorN< 4, T >( out_shape );
	}

	if( axis == 0 ) {
		T *out_ptr = output.data();
		std::size_t const stride = m_strides[0];
		T const *const self_base = this->data();
		for ( std::size_t i = 0; i < output_size; ++i ) {
			out_ptr[i] = static_cast<T>(
			    std::accumulate( self_base + i * stride, self_base + ( i + 1 ) * stride, 0.0 ) );
		}
	} else {
		// todo: implement this
	}
}

template< std::size_t rank, typename T >
void multiply2DTensor( TensorN< rank, T > const &tensor_0, TensorN< rank, T > const &tensor_1, TensorN< rank, T > &output )
{
	if( std::any_of( tensor_0.shape().rbegin() + 2, tensor_0.shape().rend(), []( std::size_t shape ) { return shape != 1; } ) ) {
		throw std::invalid_argument( "multiply2DTensor: tensor_0 is not a 2D tensor" );
	}
	if( std::any_of( tensor_1.shape().rbegin() + 2, tensor_1.shape().rend(), []( std::size_t shape ) { return shape != 1; } ) ) {
		throw std::invalid_argument( "multiply2DTensor: tensor_1 is not a 2D tensor" );
	}
	
	if( *( output.shape().rbegin() + 1 ) != *( tensor_0.shape().rbegin() + 1 ) || *( output.shape().rbegin() ) != *( tensor_1.shape().rbegin() ) ) {
		output.throw_if_non_owning_realloc( "multiply2DTensor: cannot reallocate non-owning output" );
		std::array< std::size_t, rank > output_shape = tensor_0.shape();
		output_shape[rank - 2] = *( output.shape().rbegin() + 1 );
		output_shape[rank - 1] = *( output.shape().rbegin() );
		output = TensorN< rank, T >( output_shape );
	}
	
	T* tensor_0_data = tensor_0.data();
	T* tensor_1_data = tensor_1.data();
	T* output_data = output.data();
	Eigen::Map< const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > > tensor_0_matrix( tensor_0_data, *( tensor_0.shape().rbegin() + 1 ), *( tensor_0.shape().rbegin() ) );
	Eigen::Map< const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > > tensor_1_matrix( tensor_1_data, *( tensor_1.shape().rbegin() + 1 ), *( tensor_1.shape().rbegin() ) );
	Eigen::Map< Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > > output_matrix( output_data, *( output.shape().rbegin() + 1 ), *( output.shape().rbegin() ) );
	
	output_matrix.noalias() = tensor_0_matrix * tensor_1_matrix;
}

template< std::size_t rank, typename T >
void TensorN< rank, T >::col2Im( std::array< std::size_t, 4 > const &kernel_shape, std::array< std::size_t, 4 > original_shape, TensorN< 4, T > &output )
	requires ( rank == 4 )
{
	// *this is the im2col matrix (rows = C_in*Kh*Kw, cols = N*H_out*W_out); layout matches im2Col().
	if( output.shape() != original_shape ) {
		output.throw_if_non_owning_realloc( "TensorN::col2Im: cannot reallocate non-owning output" );
		output = TensorN< 4, T >( original_shape, 0. );
	} else {
		std::fill( output.data(), output.data() + output.size(), static_cast< T >( 0 ) );
	}
	std::size_t const padding_i = kernel_shape[2] / 2;
	std::size_t const padding_j = kernel_shape[3] / 2;
	std::size_t const batch = original_shape[0];
	std::size_t const channel = original_shape[1];
	std::size_t const height = original_shape[2];
	std::size_t const width = original_shape[3];
	std::size_t const out_h = height - 2 * padding_i;
	std::size_t const out_w = width - 2 * padding_j;
	std::size_t const col_stride = m_shape[1];

	T *output_data = output.data();

	#ifdef _OPENMP
	#pragma omp parallel for schedule( static )
	#endif
	for ( std::size_t b = 0; b < batch; ++b ) {
		for ( std::size_t im_i = padding_i; im_i + padding_i < height; ++im_i ) {
			for ( std::size_t im_j = padding_j; im_j + padding_j < width; ++im_j ) {
				std::size_t const col_2d_index =
				    b * out_h * out_w + ( im_i - padding_i ) * out_w + ( im_j - padding_j );
				for ( std::size_t c = 0; c < channel; ++c ) {
					for ( std::size_t i = 0; i < kernel_shape[2]; ++i ) {
						for ( std::size_t j = 0; j < kernel_shape[3]; ++j ) {
							std::size_t const im_index = output.index(
							    { b, c, im_i - padding_i + i, im_j - padding_j + j } );
							std::size_t const row = c * kernel_shape[2] * kernel_shape[3] +
							                        i * kernel_shape[3] + j;
							output_data[im_index] += data()[row * col_stride + col_2d_index];
						}
					}
				}
			}
		}
	}
}

template< std::size_t rank, typename T >
void TensorN< rank, T >::addElementwise( std::array< std::size_t, rank > const &indices, value_type value )
{
	data()[index( indices )] += value;
}

template< std::size_t rank, typename T >
TensorN<rank, T> &TensorN<rank, T>::operator*=( TensorN const &other )
{
	std::size_t const n = size();
	for ( std::size_t i = 0; i < n; ++i ) {
		data()[i] *= other.data()[i];
	}
	return *this;
}

template< std::size_t rank, typename T >
void TensorN<rank, T>::elementwiseMultiply( TensorN const &other, TensorN &out ) const
{
	if ( m_shape != other.m_shape ) {
		throw std::invalid_argument( "TensorN::elementwiseMultiply(): shape mismatch" );
	}
	if ( this == &out ) {
		const_cast<TensorN &>( *this ) *= other;
		return;
	}
	if ( out.m_shape != m_shape ) {
		out.throw_if_non_owning_realloc(
		    "TensorN::elementwiseMultiply: cannot reallocate non-owning output" );
		out = TensorN( m_shape );
	}
	#ifdef _OPENMP
	#pragma omp parallel for schedule( static )
	#endif
	for ( std::size_t i = 0; i < size(); ++i ) {
		out.data()[i] = data()[i] * other.data()[i];
	}
}

template< std::size_t rank, typename T >
TensorN<rank, T> &TensorN<rank, T>::operator*=( value_type scalar )
{
	#ifdef _OPENMP
	#pragma omp parallel for schedule( static )
	#endif
	for ( std::size_t i = 0; i < size(); ++i ) {
		data()[i] *= scalar;
	}
	return *this;
}

template< std::size_t rank, typename T >
T &TensorN<rank, T>::at( std::size_t n, std::size_t c, std::size_t h, std::size_t w ) requires ( rank == 4 )
{
	return data()[n * m_shape[1] * m_shape[2] * m_shape[3]
	            + c * m_shape[2] * m_shape[3]
	            + h * m_shape[3]
	            + w];
}

template< std::size_t rank, typename T >
T const &TensorN<rank, T>::at( std::size_t n, std::size_t c, std::size_t h, std::size_t w ) const requires ( rank == 4 )
{
	return data()[n * m_shape[1] * m_shape[2] * m_shape[3]
	            + c * m_shape[2] * m_shape[3]
	            + h * m_shape[3]
	            + w];
}

template< std::size_t rank, typename T >
void TensorN<rank, T>::addColorChannelInPlace( TensorN<1, T> const &bias ) requires ( rank == 4 )
{
	std::size_t const H = m_shape[2];
	std::size_t const W = m_shape[3];
	#ifdef _OPENMP
	#pragma omp parallel for schedule( static )
	#endif
	for ( std::size_t n = 0; n < m_shape[0]; ++n ) {
		for ( std::size_t c = 0; c < m_shape[1]; ++c ) {
			T const b = bias.data()[c];
			for ( std::size_t h = 0; h < H; ++h ) {
				for ( std::size_t w = 0; w < W; ++w ) {
					data()[n * m_shape[1] * H * W + c * H * W + h * W + w] += b;
				}
			}
		}
	}
}

template< std::size_t rank, typename T >
void TensorN<rank, T>::addColorChannelInPlace( TensorN<4, T> const &bias ) requires ( rank == 4 )
{
	std::array< std::size_t, 4 > const bs = bias.shape();
	if ( bs[0] != 1 || bs[2] != 1 || bs[3] != 1 || bs[1] != m_shape[1] ) {
		throw std::invalid_argument( "addColorChannelInPlace: expected bias shape {1, C, 1, 1} with C == input C" );
	}
	std::size_t const H = m_shape[2];
	std::size_t const W = m_shape[3];
	T const *const bd = bias.data();
	#ifdef _OPENMP
	#pragma omp parallel for schedule( static )
	#endif
	for ( std::size_t n = 0; n < m_shape[0]; ++n ) {
		for ( std::size_t c = 0; c < m_shape[1]; ++c ) {
			T const b = bd[c];
			for ( std::size_t h = 0; h < H; ++h ) {
				for ( std::size_t w = 0; w < W; ++w ) {
					data()[n * m_shape[1] * H * W + c * H * W + h * W + w] += b;
				}
			}
		}
	}
}

template< std::size_t rank, typename T >
TensorN<rank, T> &TensorN<rank, T>::cwiseGreaterInPlace( TensorN const &other, value_type scalar )
{
	if ( m_shape != other.m_shape ) {
		if ( !m_own ) {
			throw std::invalid_argument(
			    "TensorN::cwiseGreaterInPlace: non-owning tensor cannot change shape" );
		}
		m_shape = other.m_shape;
		m_strides = other.m_strides;
		m_data.resize( other.size() );
		m_view = nullptr;
	}
	
	#ifdef _OPENMP
	#pragma omp parallel for schedule( static )
	#endif
	for ( std::size_t i = 0; i < size(); ++i ) {
		data()[i] = other.data()[i] > scalar ? static_cast<T>( 1 ) : static_cast<T>( 0 );
	}

	return *this;
}

template< std::size_t rank, typename T >
TensorN<rank, T> &TensorN<rank, T>::mulNSubstractInPlace( TensorN const &other, value_type scalar )
{
	#ifdef _OPENMP
	#pragma omp parallel for schedule( static )
	#endif
	for ( std::size_t i = 0; i < size(); ++i ) {
		data()[i] -= other.data()[i] * scalar;
	}

	return *this;
}

/// NCHW conv backward for the im2col + GEMM forward path (matches \c im2ColConvolution on CPU).
///
/// \param grad_wrt_output_nchw  \(\partial L / \partial y\). \c LayerBase::getGradInput().
///        Non-const because \c swapAxes is not \c const (data is only read).
/// \param weights \(W\) \code {C_out, C_in, K_h, K_w} \endcode (only read).
/// \param im2col_from_forward  Column matrix from forward \c im2ColConvolution.
/// \param input_shape_nchw     \code {N, C_in, H, W} \endcode.
/// \param grad_weights         \(\partial L / \partial W\) (overwritten).
/// \param grad_bias_rank4      \(\partial L / \partial b\), shape \code {1, C_out, 1, 1} \endcode.
/// \param grad_wrt_input_nchw  \(\partial L / \partial x\). \c LayerBase::getGradOutput().
/// \param workspace_cnhw       Scratch 4D tensor (reused and reshaped).
template <typename T>
void convolutional_backward_im2col( TensorN<4, T> &grad_wrt_output_nchw, TensorN<4, T> &weights,
                                    TensorN<2, T> const &im2col_from_forward,
                                    std::array<std::size_t, 4> input_shape_nchw, TensorN<4, T> &grad_weights,
                                    TensorN<4, T> &grad_bias_rank4, TensorN<4, T> &grad_wrt_input_nchw,
                                    TensorN<4, T> &workspace_cnhw )
{
	std::array<std::size_t, 4> const w_shape = weights.shape();

	grad_wrt_output_nchw.swapAxes( { 1, 0, 2, 3 }, workspace_cnhw );

	workspace_cnhw.multiply( im2col_from_forward, true, grad_weights );

	workspace_cnhw.reduceSumToDim( 0, grad_bias_rank4 );

	std::array<std::size_t, 4> const grad_shape = workspace_cnhw.shape();
	TensorN<2, T> const im2col_dy_flat(
	    { grad_shape[0], grad_shape[1] * grad_shape[2] * grad_shape[3] }, workspace_cnhw.data(),
	    workspace_cnhw.data() + workspace_cnhw.size() );

	weights.swapAxes( { 1, 2, 3, 0 }, workspace_cnhw );

	workspace_cnhw.reshape(
	    { w_shape[1] * w_shape[2] * w_shape[3], w_shape[0], 1, 1 } );

	TensorN<4, T> grad_col(
	    { w_shape[1] * w_shape[2] * w_shape[3], im2col_from_forward.shape()[1], 1, 1 } );
	workspace_cnhw.multiply( im2col_dy_flat, false, grad_col );

	grad_col.col2Im( w_shape, input_shape_nchw, grad_wrt_input_nchw );
}

/// NCHW max-pool forward (valid pooling, no padding). \p output and \p argmax must already have shape
/// \code {N, C, H_out, W_out} \endcode with \f$H_{out} = (H - pool\_size)/stride + 1\f$ (same for W).
template <typename T>
void max_pool_forward_nchw( TensorN<4, T> const &input, TensorN<4, std::size_t> &argmax,
                            TensorN<4, T> &output, std::size_t pool_size, std::size_t stride )
{
	std::array<std::size_t, 4> const in_shape    = input.shape();
	std::array<std::size_t, 4> const output_shape = output.shape();
#ifdef _OPENMP
#pragma omp parallel for schedule( static )
#endif
	for ( std::size_t b = 0; b < in_shape[0]; ++b ) {
		for ( std::size_t c = 0; c < in_shape[1]; ++c ) {
			for ( std::size_t oh = 0; oh < output_shape[2]; ++oh ) {
				for ( std::size_t ow = 0; ow < output_shape[3]; ++ow ) {
					std::size_t const h_start = oh * stride;
					std::size_t const w_start = ow * stride;
					T max = std::numeric_limits<T>::lowest();
					for ( std::size_t k = 0; k < pool_size; ++k ) {
						for ( std::size_t l = 0; l < pool_size; ++l ) {
							if ( input.at( b, c, h_start + k, w_start + l ) > max ) {
								max                           = input.at( b, c, h_start + k, w_start + l );
								argmax.at( b, c, oh, ow ) = k * pool_size + l;
							}
						}
					}
					output.at( b, c, oh, ow ) = max;
				}
			}
		}
	}
}

/// NCHW max-pool backward. \p grad_wrt_pooled is \f$\partial L/\partial y\f$ (\c LayerBase::getGradInput());
/// \p grad_wrt_input is \f$\partial L/\partial x\f$ (\c getGradOutput()), pre-sized to \p input shape and zeroed here.
template <typename T>
void max_pool_backward_nchw( TensorN<4, T> const &grad_wrt_pooled, TensorN<4, std::size_t> const &argmax,
                             TensorN<4, T> &grad_wrt_input, std::size_t pool_size, std::size_t stride )
{
	std::array<std::size_t, 4> const pooled_shape = grad_wrt_pooled.shape();
	T *const out_data = grad_wrt_input.data();
	std::fill( out_data, out_data + grad_wrt_input.size(), static_cast<T>( 0 ) );

#ifdef _OPENMP
#pragma omp parallel for schedule( static )
#endif
	for ( std::size_t b = 0; b < pooled_shape[0]; ++b ) {
		for ( std::size_t c = 0; c < pooled_shape[1]; ++c ) {
			for ( std::size_t oh = 0; oh < pooled_shape[2]; ++oh ) {
				for ( std::size_t ow = 0; ow < pooled_shape[3]; ++ow ) {
					std::size_t const h_start = oh * stride;
					std::size_t const w_start = ow * stride;
					std::size_t const k = argmax.at( b, c, oh, ow ) / pool_size;
					std::size_t const l       = argmax.at( b, c, oh, ow ) % pool_size;
					T const val               = grad_wrt_pooled.at( b, c, oh, ow );
					grad_wrt_input.addElementwise( { b, c, h_start + k, w_start + l }, val );
				}
			}
		}
	}
}

} // namespace neural

#endif // NEURAL_IMPL_TENSOR_N_HPP
