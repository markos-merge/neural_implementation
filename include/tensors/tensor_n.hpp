#ifndef NEURAL_IMPL_TENSOR_N_HPP
#define NEURAL_IMPL_TENSOR_N_HPP

#include <algorithm>
#include <cstddef>
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
		
		TensorN( TensorN const &other );
		TensorN( TensorN &&other ) noexcept;
		TensorN &operator=( TensorN const &other );
		TensorN &operator=( TensorN &&other ) noexcept;
		~TensorN() = default;
		
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
		void assign( value_type const *data, std::size_t size );

		template< std::size_t other_rank >
		void assign( std::array< std::size_t, rank > const &from_slice, std::array< std::size_t, rank > const &to_slice, TensorN< other_rank, T > const &tensor );
	
		void reduceSumToDim( std::size_t axis, TensorN< 1, T > &output );
		/// Same sum as the rank-1 overload; result stored with shape `{1, K, 1, 1}` where \p K is the
		/// reduced extent along \p axis (only \p axis == 0 implemented).
		void reduceSumToDim( std::size_t axis, TensorN< 4, T > &output ) requires ( rank == 4 );
		

		void randomizeHe( std::size_t fan_in );
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
	recomputeStrides();
}

template < std::size_t rank, typename T >
TensorN<rank, T>::TensorN( std::array<std::size_t, rank> shape, value_type value )
	: m_shape( shape )
{
	std::size_t const size = std::accumulate( shape.begin(), shape.end(), 1, std::multiplies<>() );
	m_data.resize( size, value );
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
	recomputeStrides();
}

template < std::size_t rank, typename T >
TensorN<rank, T>::TensorN( TensorN const &other )
	: m_shape( other.m_shape )
	, m_strides( other.m_strides )
{
	m_data = other.m_data;
}

template < std::size_t rank, typename T >
TensorN<rank, T>::TensorN( TensorN &&other ) noexcept
	: m_shape( other.m_shape )
	, m_strides( other.m_strides )
{
	m_data = std::move( other.m_data );
}

template < std::size_t rank, typename T >
TensorN<rank, T> &TensorN<rank, T>::operator=( TensorN const &other )
{
	m_shape = other.m_shape;
	m_strides = other.m_strides;
	m_data = other.m_data;
	return *this;
}

template < std::size_t rank, typename T >
TensorN<rank, T> &TensorN<rank, T>::operator=( TensorN &&other ) noexcept
{
	m_shape = other.m_shape;
	m_strides = other.m_strides;
	m_data = std::move( other.m_data );
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
	return m_data.data();
}

template < std::size_t rank, typename T >
T const *TensorN<rank, T>::data() const noexcept
{
	return m_data.data();
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
	return m_data[index( indices )];
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
		output = TensorN< rank, T >( new_shape );
	}
	
	T *data = output.data();
	std::array< std::size_t, rank > &strides = output.m_strides;

	loopIndicesParallel( {}, m_shape, [this, &new_axes, &strides, &data]( std::array<std::size_t, rank> const &indices ){
		std::size_t idx = 0;
		std::size_t from_idx = 0;
			for( std::size_t i = 0; i < rank; ++i ) {
				idx += indices[new_axes[i]] * strides[i];
				from_idx += indices[i] * m_strides[i];
			}

			data[idx] = m_data[from_idx];

			return false;
	} );

}

template < std::size_t rank, typename T >
void TensorN<rank, T>::assign( std::array< std::size_t, rank > const &indices, value_type value )
{
	m_data[index( indices )] = value;
}

template < std::size_t rank, typename T >
void TensorN<rank, T>::assign( std::vector< value_type > const &data )
{
	if( data.size() != size() )
	{
		throw std::invalid_argument( "data size does not match shape" );
	}
	m_data = data;
}

template < std::size_t rank, typename T >
void TensorN<rank, T>::assign( value_type const *data, std::size_t size )
{
	if( size != this->size() )
	{
		throw std::invalid_argument( "TensorN::assign(): size does not match tensor size" );
	}
	std::copy( data, data + size, m_data.begin() );
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
					    m_data[cur_im_index];
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
		gemm_cnhw_layout = TensorN< 4, T >( cnhw_shape );
	}
	kernel.multiply( im2col_tensor, false, gemm_cnhw_layout );
	gemm_cnhw_layout.swapAxes( { 1, 0, 2, 3 }, output );
}

template< std::size_t rank, typename T >
void TensorN< rank, T >::randomizeHe( std::size_t fan_in )
{
	double const scale = std::sqrt( 2.0 / static_cast<double>( fan_in ) );
	std::random_device rd;
	std::mt19937 gen( rd() );
	std::uniform_real_distribution<double> dis( -scale, scale );
	auto generator = [&]() { return static_cast<T>( dis( gen ) ); };

	std::generate( m_data.data(), m_data.data() + m_data.size(), generator );
}

//different from reduceSum, this function reduces the tensor to a single dimension
template< std::size_t rank, typename T >
void TensorN< rank, T >::reduceSumToDim( std::size_t axis, TensorN< 1, T > &output )
{
	std::size_t const output_size = m_shape[axis];
	if( output.shape()[0] != output_size ) {
		output = TensorN< 1, T >( { output_size } );
	}
	
	if( axis == 0 ) {
		T* data = output.data();
		std::size_t const stride = m_strides[0];
		
		#if _OPENMP
		#pragma omp parallel for schedule( static )
		#endif
		for( std::size_t i = 0; i < output.size(); ++i ) {
			data[i] = std::accumulate( m_data.begin() + i*stride, m_data.begin() + (i+1)*stride, 0.0 );
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
		output = TensorN< 4, T >( out_shape );
	}

	if( axis == 0 ) {
		T *data = output.data();
		std::size_t const stride = m_strides[0];
		for ( std::size_t i = 0; i < output_size; ++i ) {
			data[i] = static_cast<T>(
			    std::accumulate( m_data.begin() + i * stride, m_data.begin() + ( i + 1 ) * stride, 0.0 ) );
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
							output_data[im_index] += m_data[row * col_stride + col_2d_index];
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
	m_data[index( indices )] += value;
}

template< std::size_t rank, typename T >
TensorN<rank, T> &TensorN<rank, T>::operator*=( TensorN const &other )
{
	for ( std::size_t i = 0; i < m_data.size(); ++i ) {
		m_data[i] *= other.m_data[i];
	}
	return *this;
}

template< std::size_t rank, typename T >
TensorN<rank, T> &TensorN<rank, T>::operator*=( value_type scalar )
{
	#ifdef _OPENMP
	#pragma omp parallel for schedule( static )
	#endif
	for ( std::size_t i = 0; i < m_data.size(); ++i ) {
		m_data[i] *= scalar;
	}
	return *this;
}

template< std::size_t rank, typename T >
T &TensorN<rank, T>::at( std::size_t n, std::size_t c, std::size_t h, std::size_t w ) requires ( rank == 4 )
{
	return m_data[n * m_shape[1] * m_shape[2] * m_shape[3]
	            + c * m_shape[2] * m_shape[3]
	            + h * m_shape[3]
	            + w];
}

template< std::size_t rank, typename T >
T const &TensorN<rank, T>::at( std::size_t n, std::size_t c, std::size_t h, std::size_t w ) const requires ( rank == 4 )
{
	return m_data[n * m_shape[1] * m_shape[2] * m_shape[3]
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
					m_data[n * m_shape[1] * H * W + c * H * W + h * W + w] += b;
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
					m_data[n * m_shape[1] * H * W + c * H * W + h * W + w] += b;
				}
			}
		}
	}
}

template< std::size_t rank, typename T >
TensorN<rank, T> &TensorN<rank, T>::cwiseGreaterInPlace( TensorN const &other, value_type scalar )
{
	if ( m_shape != other.m_shape ) {
		m_shape = other.m_shape;
		m_strides = other.m_strides;
		m_data.resize( other.m_data.size() );
	}
	
	#ifdef _OPENMP
	#pragma omp parallel for schedule( static )
	#endif
	for ( std::size_t i = 0; i < m_data.size(); ++i ) {
		m_data[i] = other.m_data[i] > scalar ? static_cast<T>( 1 ) : static_cast<T>( 0 );
	}

	return *this;
}

template< std::size_t rank, typename T >
TensorN<rank, T> &TensorN<rank, T>::mulNSubstractInPlace( TensorN const &other, value_type scalar )
{
	#ifdef _OPENMP
	#pragma omp parallel for schedule( static )
	#endif
	for ( std::size_t i = 0; i < m_data.size(); ++i ) {
		m_data[i] -= other.m_data[i] * scalar;
	}

	return *this;
}

} // namespace neural

#endif // NEURAL_IMPL_TENSOR_N_HPP
