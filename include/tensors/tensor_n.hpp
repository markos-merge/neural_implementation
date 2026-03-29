#ifndef NEURAL_IMPL_TENSOR_N_HPP
#define NEURAL_IMPL_TENSOR_N_HPP

#include <cstddef>
#include <limits>
#include <vector>
#include <iterator>
#include <numeric>
#include <array>
#include <Eigen/Dense>

namespace neural {

template <typename T = float>
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
	
		TensorN() noexcept = default;
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
		std::size_t size() const noexcept;
	
		value_type operator()( std::array< std::size_t, rank > const &indices ) const;
	
		void assign( std::array< std::size_t, rank > const &indices, value_type value );
		void assign( std::vector< value_type > const &data );

		template< std::size_t other_rank >
		void assign( std::array< std::size_t, rank > const &from_slice, std::array< std::size_t, rank > const &to_slice, TensorN< other_rank, T > const &tensor );
		

		// im2col convolution: input (*this), kernel, workspace, and output are all rank-4 (e.g. NCHW).
		void im2Col( std::array< std::size_t, 3 > const &kernel_shape, TensorN< 2, T > &output ) const requires ( rank == 4 );
		void im2ColConvolution( TensorN< 4, T > const &kernel, TensorN< 2, T > &im2col_tensor, TensorN< 4, T > &output )
			requires ( rank == 4 );
	protected:
		std::vector<value_type> m_data;
	private:
		std::size_t index( std::array<std::size_t, rank> const indices ) const;

		template < typename Op >
		bool loopIndices( std::array<std::size_t, rank> const &from_indices, std::array< std::size_t, rank > const &to_indices, Op op );

	private:
		std::array<std::size_t, rank> m_shape;
};

template < std::size_t rank, typename T >
TensorN<rank, T>::TensorN( std::array<std::size_t, rank> shape )
	: m_shape( shape )
{
	std::size_t const size = std::accumulate( shape.begin(), shape.end(), 1, std::multiplies<>() );
	m_data.resize( size, 0. );
}

template < std::size_t rank, typename T >
TensorN<rank, T>::TensorN( std::array<std::size_t, rank> shape, value_type value )
	: m_shape( shape )
{
	std::size_t const size = std::accumulate( shape.begin(), shape.end(), 1, std::multiplies<>() );
	m_data.resize( size, value );
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
}

template < std::size_t rank, typename T >
TensorN<rank, T>::TensorN( TensorN const &other )
	: m_shape( other.m_shape )
{
	m_data = other.m_data;
}

template < std::size_t rank, typename T >
TensorN<rank, T>::TensorN( TensorN &&other ) noexcept
	: m_shape( other.m_shape )
{
	m_data = std::move( other.m_data );
}

template < std::size_t rank, typename T >
TensorN<rank, T> &TensorN<rank, T>::operator=( TensorN const &other )
{
	m_shape = other.m_shape;
	m_data = other.m_data;
	return *this;
}

template < std::size_t rank, typename T >
TensorN<rank, T> &TensorN<rank, T>::operator=( TensorN &&other ) noexcept
{
	m_shape = other.m_shape;
	m_data = std::move( other.m_data );
	return *this;
}

template < std::size_t rank, typename T >
std::array<std::size_t, rank> TensorN<rank, T>::shape() const noexcept
{
	return m_shape;
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
T TensorN<rank, T>::operator()( std::array<std::size_t, rank> const &indices ) const
{
	return m_data[index( indices )];
}

template < std::size_t rank, typename T >
std::size_t TensorN<rank, T>::index( std::array<std::size_t, rank> const indices ) const
{
	std::size_t index = 0;

	for( std::size_t i = 0; i < rank; ++i ) {
		index += indices[i] * std::accumulate( m_shape.begin() + i + 1, m_shape.end(),
		                                       1, std::multiplies<>() );
	}

	return index;
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

namespace detail {

template < std::size_t Rank >
std::array< std::size_t, Rank > linearToRowMajorIndices( std::size_t linear,
                                                         std::array< std::size_t, Rank > const &shape )
{
	std::array< std::size_t, Rank > out{};
	for ( std::size_t d = 0; d < Rank; ++d ) {
		std::size_t const stride = std::accumulate( shape.begin() + static_cast< std::ptrdiff_t >( d ) + 1,
		                                            shape.end(), 1, std::multiplies<>() );
		out[d] = linear / stride;
		linear %= stride;
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

} // namespace detail

template < std::size_t rank, typename T >
template < typename Op >
bool TensorN<rank, T>::loopIndices( std::array<std::size_t, rank> const &from_indices, std::array< std::size_t, rank > const &to_indices, Op op )
{
	auto cur_index = std::array<std::size_t, rank>{};
	return detail::loopIndices< Op, rank, T, 0 >( std::forward<Op>( op ), *this, from_indices, to_indices, cur_index );
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

	std::size_t linear = 0;
	loopIndices( from_slice, to_slice, [&linear, this, &tensor]( std::array<std::size_t, rank> const indices ){
		std::array<std::size_t, other_rank> const other_indices =
		    detail::linearToRowMajorIndices<other_rank>( linear++, tensor.shape() );
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

	for ( std::size_t kernel_channel = 0; kernel_channel < kernel_shape[0];
	      ++kernel_channel ) {
		for ( std::size_t kernel_row = 0; kernel_row < kernel_shape[1];
		      ++kernel_row ) {
			for ( std::size_t kernel_col = 0; kernel_col < kernel_shape[2];
			      ++kernel_col ) {
				for ( std::size_t b = 0; b < m_shape[0]; ++b ) { // batches number
					for ( std::size_t i = padding_i; i + padding_i < m_shape[2]; ++i ) {
						for ( std::size_t j = padding_j; j + padding_j < m_shape[3]; ++j ) {
							std::size_t const output_col =
							    ( b*( m_shape[2] - 2*padding_i )*(m_shape[3] - 2*padding_j ) + ( i - padding_i ) * ( m_shape[3] - 2*padding_j ) + ( j - padding_j ) );
							std::size_t const cur_row =
							    ( kernel_channel * kernel_shape[2] * kernel_shape[1] +
							      kernel_row * kernel_shape[2] + kernel_col );
							std::size_t const out_index =
							    cur_row * output_col_stride + output_col;
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
	}
}

template <std::size_t rank, typename T >
void TensorN<rank, T>::im2ColConvolution(
	TensorN< 4, T > const &kernel,
	TensorN< 2, T > &im2col_tensor,
	TensorN< 4, T > &output )
	requires ( rank == 4 )
{

	std::array< std::size_t, 3 > const kernel_shape{ kernel.shape()[1], kernel.shape()[2], kernel.shape()[3] };
	im2Col( kernel_shape, im2col_tensor );
	
	std::size_t const padding_i = kernel.shape()[2] / 2;
	std::size_t const padding_j = kernel.shape()[3] / 2;
	std::array< std::size_t, 4 > const output_shape =  std::array<std::size_t, 4>{ m_shape[0], kernel.shape()[0], ( m_shape[2] - 2*padding_i ), ( m_shape[3] - 2*padding_j ) };
	if( output.shape() != output_shape ) {
		output = TensorN< 4, T >( output_shape );
	}

	Eigen::Map< const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > > kernel_matrix(
	    kernel.data(), kernel.shape()[0], kernel.shape()[1] * kernel.shape()[2] * kernel.shape()[3] );
	Eigen::Map<Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> aux_tensor_matrix( im2col_tensor.data(), im2col_tensor.shape()[0], im2col_tensor.shape()[1] );
	Eigen::Map<Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> output_matrix( output.data(), output.shape()[1], output.shape()[0]*output.shape()[2]*output.shape()[3] );
	output_matrix.noalias() = kernel_matrix * aux_tensor_matrix;
}

} // namespace neural

#endif // NEURAL_IMPL_TENSOR_N_HPP
