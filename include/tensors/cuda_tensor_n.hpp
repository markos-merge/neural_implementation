#ifndef NEURAL_IMPL_CUDA_TENSOR_N_HPP
#define NEURAL_IMPL_CUDA_TENSOR_N_HPP

#include "cuda_mem.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <cudnn_ops.h>
#include <functional>
#include <iterator>
#include <random>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#if NEURAL_CUDNN_ENABLED

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "cublas_handle.hpp"
#include "cudnn_handle.hpp"
#include "neural_cuda_kernels.hpp"
#include "tensor_n.hpp"

namespace neural {

// =============================================================================
// detail — RAII wrappers for cuDNN descriptors
// =============================================================================

namespace detail {

struct CudnnTensorDesc
{
	cudnnTensorDescriptor_t desc{};
	CudnnTensorDesc()  { cudnnCreateTensorDescriptor( &desc ); }
	~CudnnTensorDesc() { cudnnDestroyTensorDescriptor( desc ); }
	CudnnTensorDesc( CudnnTensorDesc const & ) = delete;
	CudnnTensorDesc &operator=( CudnnTensorDesc const & ) = delete;
};

struct CudnnFilterDesc
{
	cudnnFilterDescriptor_t desc{};
	CudnnFilterDesc()  { cudnnCreateFilterDescriptor( &desc ); }
	~CudnnFilterDesc() { cudnnDestroyFilterDescriptor( desc ); }
	CudnnFilterDesc( CudnnFilterDesc const & ) = delete;
	CudnnFilterDesc &operator=( CudnnFilterDesc const & ) = delete;
};

struct CudnnConvDesc
{
	cudnnConvolutionDescriptor_t desc{};
	CudnnConvDesc()  { cudnnCreateConvolutionDescriptor( &desc ); }
	~CudnnConvDesc() { cudnnDestroyConvolutionDescriptor( desc ); }
	CudnnConvDesc( CudnnConvDesc const & ) = delete;
	CudnnConvDesc &operator=( CudnnConvDesc const & ) = delete;
};

struct CudnnReduceDesc
{
	cudnnReduceTensorDescriptor_t desc{};
	CudnnReduceDesc()  { cudnnCreateReduceTensorDescriptor( &desc ); }
	~CudnnReduceDesc() { cudnnDestroyReduceTensorDescriptor( desc ); }
	CudnnReduceDesc( CudnnReduceDesc const & ) = delete;
	CudnnReduceDesc &operator=( CudnnReduceDesc const & ) = delete;
};

struct CudnnPoolDesc
{
	cudnnPoolingDescriptor_t desc{};
	CudnnPoolDesc()  { cudnnCreatePoolingDescriptor( &desc ); }
	~CudnnPoolDesc() { cudnnDestroyPoolingDescriptor( desc ); }
	CudnnPoolDesc( CudnnPoolDesc const & ) = delete;
	CudnnPoolDesc &operator=( CudnnPoolDesc const & ) = delete;
};

// Row-major strides for `shape` (same convention as TensorN).
template <std::size_t Rank>
std::array<std::size_t, Rank> ntensor_row_major_strides( std::array<std::size_t, Rank> const &shape )
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

// Decode `linear` using row-major strides `strides` (matches TensorN::linearToRowMajorIndices).
template <std::size_t Rank>
std::array<std::size_t, Rank> ntensor_linear_to_row_major_indices(
    std::size_t linear, std::array<std::size_t, Rank> const &strides )
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

template <std::size_t Rank>
std::size_t ntensor_flat_index_from_strides( std::array<std::size_t, Rank> const &indices,
                                             std::array<std::size_t, Rank> const &strides )
{
	std::size_t idx = 0;
	for ( std::size_t i = 0; i < Rank; ++i ) {
		idx += indices[i] * strides[i];
	}
	return idx;
}

} // namespace detail

// =============================================================================
// CudaTensorNBase — device memory management shared by all ranks
// =============================================================================

template <std::size_t rank, typename T = float>
class CudaTensorNBase
{
	public:
		using value_type = T;
		static constexpr std::size_t rank_v = rank;

		// ----- constructors / destructor ------------------------------------

		CudaTensorNBase() noexcept = default;
		explicit CudaTensorNBase( std::array<std::size_t, rank> shape );
		CudaTensorNBase( std::array<std::size_t, rank> shape, T value );

		template <std::random_access_iterator It>
		CudaTensorNBase( std::array<std::size_t, rank> shape, It begin, It end );

		CudaTensorNBase( CudaTensorNBase const &other );
		CudaTensorNBase( CudaTensorNBase &&other ) noexcept;
		CudaTensorNBase &operator=( CudaTensorNBase const &other );
		CudaTensorNBase &operator=( CudaTensorNBase &&other ) noexcept;
		~CudaTensorNBase();

		// ----- accessors ----------------------------------------------------

		/// Returns a raw device pointer. Never dereference on the host.
		T       *data() noexcept       { return m_data; }
		T const *data() const noexcept { return m_data; }

		std::array<std::size_t, rank> shape()   const noexcept { return m_shape;   }
		std::array<std::size_t, rank> strides() const noexcept { return m_strides; }
		std::size_t size() const noexcept;

		/// Single-element read — copies one value device→host. Slow; debug only.
		T operator()( std::array<std::size_t, rank> const &indices ) const;

		// ----- shape mutation -----------------------------------------------

		/// Update shape/strides without moving data. Total size must stay the same.
		void reshape( std::array<std::size_t, rank> const &new_shape );

		// ----- bulk host↔device transfer ------------------------------------

		void assign( std::vector<T> const &data );
		void assign( T const *data, std::size_t size ); // cudaMemcpyDefault

		// ----- single-element mutation (slow — device↔host round-trip) -----

		void assign( std::array<std::size_t, rank> const &indices, T value );
		void addElementwise( std::array<std::size_t, rank> const &indices, T value );

		// ----- slice-based assign (sub-block between tensors of any rank) ---

		template <std::size_t other_rank>
		void assign( std::array<std::size_t, rank> const &from_slice,
		             std::array<std::size_t, rank> const &to_slice,
		             CudaTensorNBase<other_rank, T> const &tensor );

		// ----- reductions ---------------------------------------------------

		void reduceSumToDim( std::size_t axis, CudaTensorNBase<1, T> &output );

		// ----- tensor permutation (all ranks) -------------------------------

		void swapAxes( std::array<std::size_t, rank> const &new_axes,
		               CudaTensorNBase<rank, T> &output );

		// ----- element-wise ops (reuse existing cuda kernels) ---------------

		CudaTensorNBase &operator*=( CudaTensorNBase const &other );
		CudaTensorNBase &operator*=( T scalar );
		/// \p out = \c *this ⊙ \p other; (re)allocates \p out to \c m_shape if needed.
		void elementwiseMultiply( CudaTensorNBase const &other, CudaTensorNBase &out ) const;
		CudaTensorNBase &cwiseGreaterInPlace( CudaTensorNBase const &other, T scalar );
		CudaTensorNBase &mulNSubstractInPlace( CudaTensorNBase const &other, T scalar );

		// ----- misc ---------------------------------------------------------

		template <typename Generator>
		void randomize( Generator &generator );
		/// Seeded uniform \f$[0, 1)\f$; uses the \c cuda_random_uniform_float fast path for
		/// \c float. Deterministic given \p seed.
		void randomize( std::uint64_t seed );
		void randomizeHe( std::size_t fan_in, std::uint64_t seed );
		void randomizePytorchDefault( std::size_t fan_in, std::uint64_t seed );

	protected:
		T *m_data = nullptr;
		std::array<std::size_t, rank> m_shape{};
		std::array<std::size_t, rank> m_strides{};

	private:
		void recomputeStrides() noexcept;
		std::size_t flatIndex( std::array<std::size_t, rank> const &indices ) const;
};


template <typename T>
class CudaTensor4 : public CudaTensorNBase<4, T>
{
	public:
		using value_type = T;
		static constexpr std::size_t rank_v = 4;

		template <std::size_t other_rank, typename type>
		using tensor_alias = CudaTensorNBase<other_rank, type>;

		// Inherit all base constructors
		using CudaTensorNBase<4, T>::CudaTensorNBase;

		// ----------------------------------------------------------------
		// rank-4 single-element access (NCHW convenience overload)
		// ----------------------------------------------------------------

		T       &at( std::size_t n, std::size_t c, std::size_t h, std::size_t w );
		T const &at( std::size_t n, std::size_t c, std::size_t h, std::size_t w ) const;

		// ----------------------------------------------------------------
		// cuDNN — convolution forward
		//
		// Stride 1, fixed height/width padding 1 in \c cudnnSetConvolution2dDescriptor
		// (typical 3×3 "same" geometry). \p im2col_tensor is not written here. \p gemm_cnhw_layout
		// unused (API parity). CPU im2col uses valid-style geometry; paths differ unless both are
		// aligned to the same padding.
		// ----------------------------------------------------------------

		/// Intentionally not implemented on device; use another path if you need im2col.
		void im2Col( std::array<std::size_t, 3> const &kernel_shape,
		             CudaTensor4<T> &output ) const;

		/// Optional \p cudnn_workspace_cache / \p cudnn_workspace_capacity_bytes: persistent buffer reused across
		/// calls (e.g. owned by \c ConvolutionalLayer). When both are null, allocates/frees workspace per call.
		void im2ColConvolution( CudaTensor4<T> const &kernel,
		                        CudaTensorNBase<2, T> &im2col_tensor,
		                        CudaTensor4<T> &output,
		                        CudaTensor4<T> &gemm_cnhw_layout,
		                        void **cudnn_workspace_cache = nullptr,
		                        std::size_t *cudnn_workspace_capacity_bytes = nullptr );

		// ----------------------------------------------------------------
		// cuDNN — bias broadcast  {1,C,1,1} + {N,C,H,W}
		// ----------------------------------------------------------------

		void addColorChannelInPlace( CudaTensorNBase<1, T> const &bias );
		void addColorChannelInPlace( CudaTensor4<T> const &bias );

};

template <typename U>
struct is_cuda_tensor4 : std::false_type
{};
template <typename U>
struct is_cuda_tensor4<CudaTensor4<U>> : std::true_type
{};
template <typename U>
inline constexpr bool is_cuda_tensor4_v = is_cuda_tensor4<U>::value;

template <std::size_t rank, typename T>
CudaTensorNBase< rank, T >::CudaTensorNBase( std::array< std::size_t, rank > shape )
	: m_shape( shape )
{
	std::size_t const size = std::accumulate( shape.begin(), shape.end(), 1, std::multiplies<>() );
	if(detail::cuda_malloc_retry( reinterpret_cast<void **>( &m_data ), size * sizeof(T) ) != cudaSuccess) {
		throw std::runtime_error( "device memory allocation failed" );
	}

	if constexpr ( std::is_same_v<T, float> ) {
		cuda_fill_float( m_data, size, 0.f );
	} else {
		if ( cudaMemset( m_data, 0, size * sizeof( T ) ) != cudaSuccess ) {
			throw std::runtime_error( "CudaTensorNBase: cudaMemset failed" );
		}
	}

	recomputeStrides();
}

template <std::size_t rank, typename T>
CudaTensorNBase< rank, T >::CudaTensorNBase( std::array< std::size_t, rank > shape, T value )
	: m_shape( shape )
{
	std::size_t const size = std::accumulate( shape.begin(), shape.end(), 1, std::multiplies<>() );
	if(detail::cuda_malloc_retry( reinterpret_cast<void **>( &m_data ), size * sizeof(T) ) != cudaSuccess) {
		throw std::runtime_error( "device memory allocation failed" );
	}

	if constexpr ( std::is_same_v<T, float> ) {
		cuda_fill_float( m_data, size, value );
	} else {
		std::vector<T> host( size, value );
		if ( cudaMemcpy( m_data, host.data(), size * sizeof( T ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
			throw std::runtime_error( "CudaTensorNBase(shape,value): cudaMemcpy failed" );
		}
	}

	recomputeStrides();
}

template <std::size_t rank, typename T>
template <std::random_access_iterator It>
CudaTensorNBase< rank, T >::CudaTensorNBase( std::array< std::size_t, rank > shape, It begin, It end )
	: m_shape( shape )
{
	std::size_t const size = std::accumulate( shape.begin(), shape.end(), 1, std::multiplies<>() );
	if(detail::cuda_malloc_retry( reinterpret_cast<void **>( &m_data ), size * sizeof(T) ) != cudaSuccess) {
		throw std::runtime_error( "device memory allocation failed" );
	}

	std::vector<T> host( begin, end );
	if(cudaMemcpy( m_data, host.data(), size * sizeof(T), cudaMemcpyHostToDevice ) != cudaSuccess) {
		throw std::runtime_error( "host to device copy failed" );
	}

	recomputeStrides();
}

template <std::size_t rank, typename T>
CudaTensorNBase< rank, T >::~CudaTensorNBase()
{
	if(m_data != nullptr) {
		cudaFree( m_data );
	}
}

template <std::size_t rank, typename T>
CudaTensorNBase< rank, T >::CudaTensorNBase( CudaTensorNBase const &other )
	: m_shape( other.m_shape )
{
	if(detail::cuda_malloc_retry( reinterpret_cast<void **>( &m_data ), other.size() * sizeof(T) ) != cudaSuccess) {
		throw std::runtime_error( "device memory allocation failed" );
	}

	if( cudaMemcpy( m_data, other.m_data, other.size() * sizeof(T), cudaMemcpyDeviceToDevice ) != cudaSuccess ) {
		throw std::runtime_error( "device to device copy failed" );
	}

	recomputeStrides();
}

template <std::size_t rank, typename T>
CudaTensorNBase< rank, T >::CudaTensorNBase( CudaTensorNBase &&other ) noexcept
	: m_shape( other.m_shape )
{
	m_data = other.m_data;
	other.m_data = nullptr;

	recomputeStrides();
}

template <std::size_t rank, typename T>
CudaTensorNBase< rank, T > &CudaTensorNBase< rank, T >::operator=( CudaTensorNBase const &other )
{
	if(this == &other) {
		return *this;
	}

	if(m_data != nullptr) {
		cudaFree( m_data );
	}

	m_data = nullptr;
	m_shape = other.m_shape;

	if(detail::cuda_malloc_retry( reinterpret_cast<void **>( &m_data ), other.size() * sizeof(T) ) != cudaSuccess) {
		throw std::runtime_error( "device memory allocation failed" );
	}

	if(cudaMemcpy( m_data, other.m_data, other.size() * sizeof(T), cudaMemcpyDeviceToDevice ) != cudaSuccess) {
		throw std::runtime_error( "device to device copy failed" );
	}

	recomputeStrides();

	return *this;
}

template <std::size_t rank, typename T>
CudaTensorNBase< rank, T > &CudaTensorNBase< rank, T >::operator=( CudaTensorNBase &&other ) noexcept
{
	if(this == &other) {
		return *this;
	}
	
	if( m_data != nullptr ) {
		cudaFree( m_data );
	}

	m_data = other.m_data;
	m_shape = other.m_shape;
	other.m_data = nullptr;

	recomputeStrides();

	return *this;
}

template <std::size_t rank, typename T>
std::size_t CudaTensorNBase< rank, T >::size() const noexcept
{
	return std::accumulate( m_shape.begin(), m_shape.end(), 1, std::multiplies<>() );
}

template <std::size_t rank, typename T>
void CudaTensorNBase< rank, T >::recomputeStrides() noexcept
{
	if constexpr ( rank > 0 ) {
		m_strides[rank - 1] = 1;
		for ( std::size_t i = rank - 1; i > 0; --i ) {
			m_strides[i - 1] = m_strides[i] * m_shape[i];
		}
	}
}

template <std::size_t rank, typename T>
std::size_t CudaTensorNBase< rank, T >::flatIndex( std::array<std::size_t, rank> const &indices ) const
{
	std::size_t idx = 0;
	for ( std::size_t i = 0; i < rank; ++i ) {
		idx += indices[i] * m_strides[i];
	}
	return idx;
}

template <std::size_t rank, typename T>
T CudaTensorNBase< rank, T >::operator()( std::array<std::size_t, rank> const &indices ) const
{
	T val{};

	cudaMemcpy( &val, m_data + flatIndex(indices), sizeof(T), cudaMemcpyDeviceToHost );

	return val;
}

template <std::size_t rank, typename T>
void CudaTensorNBase< rank, T >::reshape( std::array<std::size_t, rank> const &new_shape )
{
	if( std::accumulate( new_shape.begin(), new_shape.end(), 1, std::multiplies<>() ) != size() ) {
		throw std::invalid_argument( "new shape size does not match tensor size" );
	}
	m_shape = new_shape;
	recomputeStrides();
}

template <std::size_t rank, typename T>
void CudaTensorNBase< rank, T >::assign( std::vector<T> const &data )
{
	if(data.size() != size()) {
		throw std::invalid_argument( "data size does not match tensor size" );
	}

	cudaMemcpy( m_data, data.data(), size() * sizeof(T), cudaMemcpyHostToDevice );
}

template <std::size_t rank, typename T>
void CudaTensorNBase< rank, T >::assign( T const *data, std::size_t size )
{
	if(size != this->size() ) {
		throw std::invalid_argument( "data size does not match tensor size" );
	}

	if ( cudaMemcpy( m_data, data, size * sizeof( T ), cudaMemcpyDefault ) != cudaSuccess ) {
		throw std::runtime_error( "CudaTensorNBase::assign(ptr): cudaMemcpy failed" );
	}
}

template <std::size_t rank, typename T>
void CudaTensorNBase< rank, T >::assign( std::array<std::size_t, rank> const &indices, T value )
{
	if ( cudaMemcpy( m_data + flatIndex( indices ), &value, sizeof( T ), cudaMemcpyHostToDevice )
	     != cudaSuccess ) {
		throw std::runtime_error( "CudaTensorNBase::assign(indices): host to device copy failed" );
	}
}

template <std::size_t rank, typename T>
void CudaTensorNBase< rank, T >::addElementwise( std::array<std::size_t, rank> const &indices, T value )
{
	std::size_t const idx = flatIndex( indices );
	T *const slot = m_data + idx;
	if ( cuda_add_elementwise( slot, slot, 1, value ) != cudaSuccess ) {
		throw std::runtime_error( "CudaTensorNBase::addElementwise: cuda_add_elementwise failed" );
	}
}

template <std::size_t rank, typename T>
template <std::size_t other_rank>
void CudaTensorNBase<rank, T>::assign( std::array<std::size_t, rank> const &from_slice,
                                       std::array<std::size_t, rank> const &to_slice,
                                       CudaTensorNBase<other_rank, T> const &tensor )
{
	static_assert( rank >= other_rank, "rank must be greater than or equal to other_rank" );

	for ( std::size_t i = 0; i < rank; ++i ) {
		if ( from_slice[i] >= to_slice[i] ) {
			throw std::invalid_argument( "from_slice must be less than to_slice" );
		}
	}

	std::array<std::size_t, rank> diff{};
	for ( std::size_t i = 0; i < rank; ++i ) {
		diff[i] = to_slice[i] - from_slice[i];
	}

	std::size_t const region_size =
	    std::accumulate( diff.begin(), diff.end(), static_cast<std::size_t>( 1 ), std::multiplies<>() );
	if ( region_size != tensor.size() ) {
		throw std::invalid_argument( "region size does not match tensor size" );
	}

	if ( region_size == 0 || m_data == nullptr || tensor.data() == nullptr ) {
		return;
	}

	std::array<std::size_t, rank> const strides_box = detail::ntensor_row_major_strides( diff );
	std::array<std::size_t, other_rank> const tensor_strides = tensor.strides();

	for ( std::size_t linear = 0; linear < region_size; ++linear ) {
		std::array<std::size_t, rank> const coord =
		    detail::ntensor_linear_to_row_major_indices( linear, strides_box );
		std::array<std::size_t, rank> dst_indices{};
		for ( std::size_t d = 0; d < rank; ++d ) {
			dst_indices[d] = from_slice[d] + coord[d];
		}
		std::array<std::size_t, other_rank> const src_indices =
		    detail::ntensor_linear_to_row_major_indices( linear, tensor_strides );

		T *const dst = m_data + flatIndex( dst_indices );
		T const *const src =
		    tensor.data()
		    + detail::ntensor_flat_index_from_strides( src_indices, tensor_strides );

		if ( cudaMemcpy( dst, src, sizeof( T ), cudaMemcpyDeviceToDevice ) != cudaSuccess ) {
			throw std::runtime_error( "CudaTensorNBase::assign(slice): device to device copy failed" );
		}
	}
}

template <std::size_t rank, typename T>
void CudaTensorNBase< rank, T >::reduceSumToDim( std::size_t axis, CudaTensorNBase<1, T> &output )
{
	if(axis >= rank) {
		throw std::invalid_argument( "axis out of range" );
	}

	std::size_t const output_size = m_shape[axis];
	if(output.shape()[0] != output_size) {
		output = CudaTensorNBase<1, T>( { output_size } );
	}
	
	T *data = output.data();

	if ( axis != 0 ) {
		std::size_t const prefix = std::accumulate(
		    m_shape.begin(), m_shape.begin() + static_cast<std::ptrdiff_t>( axis ),
		    static_cast<std::size_t>( 1 ), std::multiplies<>() );
		std::size_t const mid = m_shape[axis];
		std::size_t const suffix = m_strides[axis];

		cuda_reduce_sum_to_dim( m_data, data, prefix, mid, suffix );
	} else {
		cuda_reduce_sum_to_dim_axis_0( m_data, data, output_size, m_strides[0] );
	}
}

template <std::size_t rank, typename T>
void CudaTensorNBase< rank, T >::swapAxes( std::array<std::size_t, rank> const &new_axes,
                                           CudaTensorNBase<rank, T> &output )
{
	if( new_axes.size() != rank ) {
		throw std::invalid_argument( "new_axes size does not match rank" );
	}

	std::array<std::size_t, rank> new_shape{};
	for( std::size_t i = 0; i < rank; ++i ) {
		new_shape[i] = m_shape[new_axes[i]];
	}

	if( output.shape() != new_shape ) {
		output = CudaTensorNBase<rank, T>( new_shape );
	}
	
	T *out = output.data();
	T *in = this->data();

	cuda_swap_axes( in, out, this->size(), m_strides.data(), output.m_strides.data(), rank,
	                new_axes.data() );
}

template <std::size_t rank, typename T>
CudaTensorNBase< rank, T > &CudaTensorNBase< rank, T >::operator*=( CudaTensorNBase const &other )
{
	if ( m_shape != other.m_shape ) {
		throw std::invalid_argument( "CudaTensorNBase::operator*=: shape mismatch" );
	}

	cudaError_t err = cudaErrorInvalidValue;
	if constexpr ( std::is_same_v<T, float> ) {
		err = cuda_mul_float_inplace( m_data, other.m_data, size() );
	} else if constexpr ( std::is_same_v<T, __nv_fp8_e4m3> ) {
		err = cuda_mul_fp8_inplace( m_data, other.m_data, size() );
	} else {
		throw std::runtime_error( "CudaTensorNBase::operator*=: unsupported element type" );
	}

	if ( err != cudaSuccess ) {
		throw std::runtime_error( "CudaTensorNBase::operator*=: cuda multiply failed" );
	}
	return *this;
}

template <std::size_t rank, typename T>
void CudaTensorNBase< rank, T >::elementwiseMultiply( CudaTensorNBase const &other, CudaTensorNBase &out ) const
{
	if ( m_shape != other.m_shape ) {
		throw std::invalid_argument( "CudaTensorNBase::elementwiseMultiply: shape mismatch" );
	}
	if ( this == &out ) {
		const_cast<CudaTensorNBase &>( *this ) *= other;
		return;
	}
	if ( out.m_shape != m_shape || out.m_data == nullptr ) {
		out = CudaTensorNBase<rank, T>( m_shape );
	}
	if constexpr ( std::is_same_v<T, float> ) {
		if ( cuda_mul_float( m_data, other.m_data, out.m_data, size() ) != cudaSuccess ) {
			throw std::runtime_error( "CudaTensorNBase::elementwiseMultiply: cuda_mul_float failed" );
		}
	} else if constexpr ( std::is_same_v<T, __nv_fp8_e4m3> ) {
		if ( cuda_mul_fp8( m_data, other.m_data, out.m_data, size() ) != cudaSuccess ) {
			throw std::runtime_error( "CudaTensorNBase::elementwiseMultiply: cuda_mul_fp8 failed" );
		}
	} else {
		throw std::invalid_argument( "CudaTensorNBase::elementwiseMultiply: unsupported element type" );
	}
}

template <std::size_t rank, typename T>
CudaTensorNBase< rank, T > &CudaTensorNBase< rank, T >::operator*=( T scalar )
{
	cudaError_t err = cudaErrorInvalidValue;
	if constexpr ( std::is_same_v<T, float> ) {
		err = cuda_scale_float( m_data, size(), scalar );
	} else if constexpr ( std::is_same_v<T, __nv_fp8_e4m3> ) {
		err = cuda_scale_fp8( m_data, size(), static_cast<float>( static_cast<__half>( scalar ) ) );
	} else {
		throw std::runtime_error( "CudaTensorNBase::operator*=(scalar): unsupported element type" );
	}

	if ( err != cudaSuccess ) {
		throw std::runtime_error( "CudaTensorNBase::operator*=(scalar): cuda scale failed" );
	}
	return *this;
}

template <std::size_t rank, typename T>
CudaTensorNBase< rank, T > &CudaTensorNBase< rank, T >::cwiseGreaterInPlace( CudaTensorNBase const &other,
                                                                              T scalar )
{
	if ( m_shape != other.m_shape ) {
		*this = CudaTensorNBase( other.m_shape );
	}

	cudaError_t err = cudaErrorInvalidValue;
	if constexpr ( std::is_same_v<T, float> ) {
		err = cuda_cwise_greater_float( other.m_data, m_data, size(), scalar );
	} else if constexpr ( std::is_same_v<T, __nv_fp8_e4m3> ) {
		err = cuda_cwise_greater_fp8( other.m_data, m_data, size(), static_cast<float>( static_cast<__half>( scalar ) ) );
	} else {
		throw std::runtime_error( "CudaTensorNBase::cwiseGreaterInPlace: unsupported element type" );
	}

	if ( err != cudaSuccess ) {
		throw std::runtime_error( "CudaTensorNBase::cwiseGreaterInPlace: cuda cwise greater failed" );
	}
	return *this;
}

template <std::size_t rank, typename T>
CudaTensorNBase< rank, T > &CudaTensorNBase< rank, T >::mulNSubstractInPlace( CudaTensorNBase const &other,
                                                                               T scalar )
{
	if ( m_shape != other.m_shape ) {
		throw std::invalid_argument( "CudaTensorNBase::mulNSubstractInPlace: shape mismatch" );
	}

	if constexpr ( std::is_same_v<T, float> ) {
		if ( cuda_mul_substract_float( m_data, other.m_data, size(), scalar ) != cudaSuccess ) {
			throw std::runtime_error( "CudaTensorNBase::mulNSubstractInPlace: cuda mul-subtract failed" );
		}
	} else {
		throw std::runtime_error( "CudaTensorNBase::mulNSubstractInPlace: unsupported element type" );
	}
	return *this;
}

template <std::size_t rank, typename T>
template <typename Generator>
void CudaTensorNBase< rank, T >::randomize( Generator &generator )
{
	if ( m_data == nullptr || size() == 0 ) {
		return;
	}
	std::vector<T> host( size() );
	std::generate( host.begin(), host.end(), generator );
	assign( host );
}

template <std::size_t rank, typename T>
void CudaTensorNBase< rank, T >::randomize( std::uint64_t seed )
{
	if ( m_data == nullptr || size() == 0 ) {
		return;
	}
	if constexpr ( std::is_same_v<T, float> ) {
		(void)cuda_random_uniform_float( m_data, size(),
		                                 static_cast<unsigned long long>( seed ) );
		return;
	}
	std::mt19937_64 gen( seed );
	std::uniform_real_distribution<double> dis( 0.0, 1.0 );
	std::vector<T> host( size() );
	for ( auto &v : host ) {
		if constexpr ( std::is_same_v<T, __nv_fp8_e4m3> ) {
			v = static_cast<T>( static_cast<__half>( static_cast<float>( dis( gen ) ) ) );
		} else {
			throw std::runtime_error( "CudaTensorNBase::randomize: unsupported element type" );
		}
	}
	assign( host );
}

template <std::size_t rank, typename T>
void CudaTensorNBase< rank, T >::randomizeHe( std::size_t fan_in, std::uint64_t seed )
{
	double const scale = std::sqrt( 2.0 / static_cast<double>( fan_in ) );
	std::mt19937_64 gen( seed );
	std::uniform_real_distribution<double> dis( -scale, scale );
	std::vector<T> host( size() );
	for ( auto &v : host ) {
		if constexpr ( std::is_same_v<T, float> ) {
			v = static_cast<T>( dis( gen ) );
		} else if constexpr ( std::is_same_v<T, __nv_fp8_e4m3> ) {
			v = static_cast<T>( static_cast<__half>( static_cast<float>( dis( gen ) ) ) );
		} else {
			throw std::runtime_error( "CudaTensorNBase::randomizeHe: unsupported element type" );
		}
	}
	assign( host );
}

template <std::size_t rank, typename T>
void CudaTensorNBase< rank, T >::randomizePytorchDefault( std::size_t fan_in, std::uint64_t seed )
{
	if ( fan_in == 0 ) {
		return;
	}
	double const inv_sqrt = 1.0 / std::sqrt( static_cast<double>( fan_in ) );
	std::mt19937_64 gen( seed );
	std::uniform_real_distribution<double> dis( -inv_sqrt, inv_sqrt );
	std::vector<T> host( size() );
	for ( auto &v : host ) {
		if constexpr ( std::is_same_v<T, float> ) {
			v = static_cast<T>( dis( gen ) );
		} else if constexpr ( std::is_same_v<T, __nv_fp8_e4m3> ) {
			v = static_cast<T>( static_cast<__half>( static_cast<float>( dis( gen ) ) ) );
		} else {
			throw std::runtime_error( "CudaTensorNBase::randomizePytorchDefault: unsupported element type" );
		}
	}
	assign( host );
}

template <typename T>
void CudaTensor4<T>::im2Col( std::array<std::size_t, 3> const &kernel_shape,
                             CudaTensor4<T> &output ) const
{
	(void)kernel_shape;
	(void)output;
}

template <typename T>
void CudaTensor4<T>::im2ColConvolution( CudaTensor4<T> const &kernel,
                                        CudaTensorNBase<2, T> &im2col_tensor,
                                        CudaTensor4<T> &output,
                                        CudaTensor4<T> &gemm_cnhw_layout,
                                        void **cudnn_workspace_cache,
                                        std::size_t *cudnn_workspace_capacity_bytes )
{
	(void)gemm_cnhw_layout;
	(void)im2col_tensor;
	if constexpr ( !std::is_same_v<T, float> ) {
		throw std::runtime_error( "CudaTensor4::im2ColConvolution: only float is supported" );
	} else {
		std::array<std::size_t, 4> const ks = kernel.shape();
		std::size_t const Kh = ks[2];
		std::size_t const Kw = ks[3];
		if ( ( Kh % 2 ) == 0 || ( Kw % 2 ) == 0 ) {
			throw std::invalid_argument(
			    "CudaTensor4::im2ColConvolution: cuDNN path requires odd kernel height and width" );
		}

		cudnnHandle_t &handle = CudnnHandle::instance();
		detail::CudnnTensorDesc input_desc;
		cudnnStatus_t st = cudnnSetTensor4dDescriptor(
		    input_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		    static_cast<int>( this->shape()[0] ), static_cast<int>( this->shape()[1] ),
		    static_cast<int>( this->shape()[2] ), static_cast<int>( this->shape()[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (input) failed" );
		}

		detail::CudnnFilterDesc filter_desc;
		st = cudnnSetFilter4dDescriptor( filter_desc.desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
		                                 static_cast<int>( ks[0] ), static_cast<int>( ks[1] ),
		                                 static_cast<int>( ks[2] ), static_cast<int>( ks[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetFilter4dDescriptor failed" );
		}

		detail::CudnnConvDesc conv_desc;
		st = cudnnSetConvolution2dDescriptor( conv_desc.desc, 1, 1, 1, 1, 1, 1,
		                                        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetConvolution2dDescriptor failed" );
		}

		int n = 0, c = 0, h = 0, w = 0;
		st = cudnnGetConvolution2dForwardOutputDim( conv_desc.desc, input_desc.desc, filter_desc.desc,
		    &n, &c, &h, &w );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnGetConvolution2dForwardOutputDim failed" );
		}

		std::array<std::size_t, 4> const out_shape_sz{ static_cast<std::size_t>( n ),
			                                       static_cast<std::size_t>( c ),
			                                       static_cast<std::size_t>( h ),
			                                       static_cast<std::size_t>( w ) };
		if ( output.shape() != out_shape_sz ) {
			output = CudaTensor4<T>( out_shape_sz );
		}

		detail::CudnnTensorDesc output_desc;
		st = cudnnSetTensor4dDescriptor( output_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h,
		                                 w );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (output) failed" );
		}

		cudnnConvolutionFwdAlgo_t const algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
		std::size_t ws_size = 0;
		st = cudnnGetConvolutionForwardWorkspaceSize( handle, input_desc.desc, filter_desc.desc,
		                                              conv_desc.desc, output_desc.desc, algo,
		                                              &ws_size );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnGetConvolutionForwardWorkspaceSize failed" );
		}

		void *workspace = nullptr;
		bool const use_persistent_ws =
		    cudnn_workspace_cache != nullptr && cudnn_workspace_capacity_bytes != nullptr;
		if ( ws_size > 0 ) {
			if ( use_persistent_ws ) {
				cudaError_t const werr = detail::cudnn_workspace_ensure(
				    cudnn_workspace_cache, cudnn_workspace_capacity_bytes, ws_size );
				if ( werr != cudaSuccess ) {
					throw std::runtime_error( "im2ColConvolution: persistent workspace allocation failed" );
				}
				workspace = *cudnn_workspace_cache;
			} else if ( detail::cuda_malloc_retry( &workspace, ws_size ) != cudaSuccess ) {
				throw std::runtime_error( "im2ColConvolution: workspace allocation failed" );
			}
		}

		float alpha = 1.f;
		float beta = 0.f;
		st          = cudnnConvolutionForward( handle, &alpha, input_desc.desc, this->data(),
		                                              filter_desc.desc, kernel.data(), conv_desc.desc,
		                                              algo, workspace, ws_size, &beta, output_desc.desc,
		                                              output.data() );
		if ( !use_persistent_ws && workspace != nullptr ) {
			(void)cudaFree( workspace );
		}
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnConvolutionForward failed" );
		}
	}
}

template <typename T>
void CudaTensor4<T>::addColorChannelInPlace( CudaTensorNBase<1, T> const &bias )
{
	if constexpr ( !std::is_same_v<T, float> ) {
		throw std::runtime_error( "CudaTensor4::addColorChannelInPlace: only float is supported" );
	} else {
		if ( bias.shape()[0] != this->shape()[1] ) {
			throw std::invalid_argument(
			    "addColorChannelInPlace: bias length must match channel dimension of activations" );
		}
		int const C = static_cast<int>( bias.shape()[0] );
		cudnnHandle_t &handle = CudnnHandle::instance();
		detail::CudnnTensorDesc bias_desc;
		cudnnStatus_t st = cudnnSetTensor4dDescriptor( bias_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		                                             1, C, 1, 1 );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (bias) failed" );
		}
		detail::CudnnTensorDesc y_desc;
		st = cudnnSetTensor4dDescriptor( y_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		                                 static_cast<int>( this->shape()[0] ),
		                                 static_cast<int>( this->shape()[1] ),
		                                 static_cast<int>( this->shape()[2] ),
		                                 static_cast<int>( this->shape()[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (activations) failed" );
		}
		float alpha = 1.f;
		float beta  = 1.f;
		st          = cudnnAddTensor( handle, &alpha, bias_desc.desc, bias.data(), &beta, y_desc.desc,
		                             this->data() );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnAddTensor failed" );
		}
	}
}

template <typename T>
void CudaTensor4<T>::addColorChannelInPlace( CudaTensor4<T> const &bias )
{
	if constexpr ( !std::is_same_v<T, float> ) {
		throw std::runtime_error( "CudaTensor4::addColorChannelInPlace: only float is supported" );
	} else {
		std::array<std::size_t, 4> const bs = bias.shape();
		if ( bs[0] != 1 || bs[2] != 1 || bs[3] != 1 || bs[1] != this->shape()[1] ) {
			throw std::invalid_argument(
			    "addColorChannelInPlace: expected bias shape {1, C, 1, 1} with C == activations C" );
		}
		cudnnHandle_t &handle = CudnnHandle::instance();
		detail::CudnnTensorDesc bias_desc;
		cudnnStatus_t st = cudnnSetTensor4dDescriptor( bias_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		                                             static_cast<int>( bs[0] ), static_cast<int>( bs[1] ),
		                                             static_cast<int>( bs[2] ), static_cast<int>( bs[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (bias) failed" );
		}
		detail::CudnnTensorDesc y_desc;
		st = cudnnSetTensor4dDescriptor( y_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		                                 static_cast<int>( this->shape()[0] ),
		                                 static_cast<int>( this->shape()[1] ),
		                                 static_cast<int>( this->shape()[2] ),
		                                 static_cast<int>( this->shape()[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (activations) failed" );
		}
		float alpha = 1.f;
		float beta  = 1.f;
		st          = cudnnAddTensor( handle, &alpha, bias_desc.desc, bias.data(), &beta, y_desc.desc,
		                             this->data() );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnAddTensor failed" );
		}
	}
}

template <typename T>
void batch_norm_forward_nchw( CudaTensor4<T> const &input, CudaTensor4<T> const &gamma,
                              CudaTensor4<T> const &beta, CudaTensor4<T> &running_mean,
                              CudaTensor4<T> &running_var, CudaTensor4<T> &last_running_mean,
                              CudaTensor4<T> &last_running_var, T eps, T momentum,
                              bool training, bool is_first_forward, CudaTensor4<T> &output )
{
	if constexpr ( !std::is_same_v<T, float> ) {
		throw std::runtime_error( "batch_norm_forward_nchw: only float is supported" );
	} else {
		std::array<std::size_t, 4> const in_shape = input.shape();
		std::size_t const C = in_shape[1];
		std::array<std::size_t, 4> const param_shape{ 1, C, 1, 1 };
		if ( output.shape() != in_shape ) {
			throw std::invalid_argument( "batch_norm_forward_nchw: output shape must match input shape" );
		}
		if ( gamma.shape() != param_shape || beta.shape() != param_shape ||
		     running_mean.shape() != param_shape || running_var.shape() != param_shape ||
		     last_running_mean.shape() != param_shape || last_running_var.shape() != param_shape ) {
			throw std::invalid_argument(
			    "batch_norm_forward_nchw: expected {1, C, 1, 1} for gamma/beta/running tensors with C == input channels" );
		}

		cudnnHandle_t &handle = CudnnHandle::instance();
		detail::CudnnTensorDesc x_desc;
		cudnnStatus_t st = cudnnSetTensor4dDescriptor(
		    x_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		    static_cast<int>( in_shape[0] ), static_cast<int>( in_shape[1] ),
		    static_cast<int>( in_shape[2] ), static_cast<int>( in_shape[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (x) failed" );
		}

		detail::CudnnTensorDesc y_desc;
		st = cudnnSetTensor4dDescriptor(
		    y_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		    static_cast<int>( in_shape[0] ), static_cast<int>( in_shape[1] ),
		    static_cast<int>( in_shape[2] ), static_cast<int>( in_shape[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (y) failed" );
		}

		detail::CudnnTensorDesc bn_desc;
		st = cudnnDeriveBNTensorDescriptor( bn_desc.desc, x_desc.desc, CUDNN_BATCHNORM_SPATIAL );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnDeriveBNTensorDescriptor failed" );
		}

		float const alpha = 1.f;
		float const beta_zero = 0.f;
		if ( training || is_first_forward ) {
			CudaTensor4<T> saved_inv_variance( param_shape );
			// Same as PyTorch / cudnnBatchNormalizationForwardTraining: factor multiplies the batch term.
			double const exp_avg_factor = static_cast<double>( momentum );
			st = cudnnBatchNormalizationForwardTraining(
			    handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta_zero, x_desc.desc, input.data(),
			    y_desc.desc, output.data(), bn_desc.desc, gamma.data(), beta.data(), exp_avg_factor,
			    last_running_mean.data(), last_running_var.data(), static_cast<double>( eps ),
			    running_mean.data(), saved_inv_variance.data() );
			if ( st != CUDNN_STATUS_SUCCESS ) {
				throw std::runtime_error( "cudnnBatchNormalizationForwardTraining failed" );
			}

			cudaError_t const rv_err = cuda_bn_running_var_from_saved_inv_std_float(
			    running_var.data(), saved_inv_variance.data(), C, static_cast<float>( eps ) );
			if ( rv_err != cudaSuccess ) {
				throw std::runtime_error(
				    "batch_norm_forward_nchw: cuda_bn_running_var_from_saved_inv_std_float failed" );
			}
		} else {
			st = cudnnBatchNormalizationForwardInference(
			    handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta_zero, x_desc.desc, input.data(),
			    y_desc.desc, output.data(), bn_desc.desc, gamma.data(), beta.data(),
			    last_running_mean.data(), last_running_var.data(), static_cast<double>( eps ) );
			if ( st != CUDNN_STATUS_SUCCESS ) {
				throw std::runtime_error( "cudnnBatchNormalizationForwardInference failed" );
			}
			running_mean = last_running_mean;
			running_var = last_running_var;
		}
	}
}

template <typename T>
void batch_norm_backward_nchw( CudaTensor4<T> const &grad_wrt_output,
                               CudaTensor4<T> const &input, CudaTensor4<T> const &gamma,
                               CudaTensor4<T> const &running_mean,
                               CudaTensor4<T> const &running_var, T eps,
                               CudaTensor4<T> &grad_gamma, CudaTensor4<T> &grad_beta,
                               CudaTensor4<T> &grad_wrt_input )
{
	if constexpr ( !std::is_same_v<T, float> ) {
		throw std::runtime_error( "batch_norm_backward_nchw: only float is supported" );
	} else {
		std::array<std::size_t, 4> const in_shape = input.shape();
		std::size_t const C = in_shape[1];
		std::array<std::size_t, 4> const param_shape{ 1, C, 1, 1 };
		if ( grad_wrt_output.shape() != in_shape || grad_wrt_input.shape() != in_shape ) {
			throw std::invalid_argument(
			    "batch_norm_backward_nchw: grad input/output shapes must match input shape" );
		}
		if ( gamma.shape() != param_shape || running_mean.shape() != param_shape ||
		     running_var.shape() != param_shape || grad_gamma.shape() != param_shape ||
		     grad_beta.shape() != param_shape ) {
			throw std::invalid_argument(
			    "batch_norm_backward_nchw: expected {1, C, 1, 1} for gamma/running/grad tensors with C == input channels" );
		}

		CudaTensor4<T> saved_inv_variance( param_shape );
		cudaError_t const si_err = cuda_bn_saved_inv_std_from_variance_float(
		    saved_inv_variance.data(), running_var.data(), C, static_cast<float>( eps ) );
		if ( si_err != cudaSuccess ) {
			throw std::runtime_error(
			    "batch_norm_backward_nchw: cuda_bn_saved_inv_std_from_variance_float failed" );
		}

		cudnnHandle_t &handle = CudnnHandle::instance();
		detail::CudnnTensorDesc x_desc;
		cudnnStatus_t st = cudnnSetTensor4dDescriptor(
		    x_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		    static_cast<int>( in_shape[0] ), static_cast<int>( in_shape[1] ),
		    static_cast<int>( in_shape[2] ), static_cast<int>( in_shape[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (x) failed" );
		}

		detail::CudnnTensorDesc dy_desc;
		st = cudnnSetTensor4dDescriptor(
		    dy_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		    static_cast<int>( in_shape[0] ), static_cast<int>( in_shape[1] ),
		    static_cast<int>( in_shape[2] ), static_cast<int>( in_shape[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (dy) failed" );
		}

		detail::CudnnTensorDesc dx_desc;
		st = cudnnSetTensor4dDescriptor(
		    dx_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		    static_cast<int>( in_shape[0] ), static_cast<int>( in_shape[1] ),
		    static_cast<int>( in_shape[2] ), static_cast<int>( in_shape[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (dx) failed" );
		}

		detail::CudnnTensorDesc bn_desc;
		st = cudnnDeriveBNTensorDescriptor( bn_desc.desc, x_desc.desc, CUDNN_BATCHNORM_SPATIAL );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnDeriveBNTensorDescriptor failed" );
		}

		float const alpha_data_diff = 1.f;
		float const beta_data_diff = 0.f;
		float const alpha_param_diff = 1.f;
		float const beta_param_diff = 0.f;
		st = cudnnBatchNormalizationBackward(
		    handle, CUDNN_BATCHNORM_SPATIAL, &alpha_data_diff, &beta_data_diff,
		    &alpha_param_diff, &beta_param_diff, x_desc.desc, input.data(), dy_desc.desc,
		    grad_wrt_output.data(), dx_desc.desc, grad_wrt_input.data(), bn_desc.desc,
		    gamma.data(), grad_gamma.data(), grad_beta.data(), static_cast<double>( eps ),
		    running_mean.data(), saved_inv_variance.data() );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnBatchNormalizationBackward failed" );
		}
	}
}

/// cuDNN conv backward (matches forward: padding 1,1, stride 1, \c CUDNN_CROSS_CORRELATION, odd \f$K_h,K_w\f$).
/// \param grad_wrt_output_nchw  \(\partial L/\partial y\) (\c LayerBase::getGradInput()).
/// \param weights               \(W\) (read-only).
/// \param input_activations     \(x\) from forward (\c getInput()).
/// \param grad_weights          \(\partial L/\partial W\) (overwritten).
/// \param grad_bias_rank4       \(\partial L/\partial b\), shape \c {1,C_out,1,1}.
/// \param grad_wrt_input_nchw \(\partial L/\partial x\) (\c getGradOutput()), resized to \p input_activations.shape() if needed.
template <typename T>
void convolutional_backward_cudnn( CudaTensor4<T> &grad_wrt_output_nchw, CudaTensor4<T> &weights,
                                   CudaTensor4<T> const &input_activations, CudaTensor4<T> &grad_weights,
                                   CudaTensor4<T> &grad_bias_rank4, CudaTensor4<T> &grad_wrt_input_nchw,
                                   void **cudnn_workspace_cache = nullptr,
                                   std::size_t *cudnn_workspace_capacity_bytes = nullptr )
{
	if constexpr ( !std::is_same_v<T, float> ) {
		throw std::runtime_error( "convolutional_backward_cudnn: only float is supported" );
	} else {
		std::array<std::size_t, 4> const ks = weights.shape();
		if ( ( ks[2] % 2 ) == 0 || ( ks[3] % 2 ) == 0 ) {
			throw std::invalid_argument(
			    "convolutional_backward_cudnn: odd kernel height and width required (same as forward)" );
		}
		if ( grad_weights.shape() != ks ) {
			throw std::invalid_argument(
			    "convolutional_backward_cudnn: grad_weights shape must match weights" );
		}
		std::array<std::size_t, 4> const bs = grad_bias_rank4.shape();
		if ( bs[0] != 1 || bs[2] != 1 || bs[3] != 1 || bs[1] != ks[0] ) {
			throw std::invalid_argument(
			    "convolutional_backward_cudnn: grad_bias must be {1, C_out, 1, 1}" );
		}
		if ( grad_wrt_input_nchw.shape() != input_activations.shape() ) {
			grad_wrt_input_nchw = CudaTensor4<T>( input_activations.shape() );
		}

		cudnnHandle_t &handle = CudnnHandle::instance();
		detail::CudnnTensorDesc x_desc;
		cudnnStatus_t st = cudnnSetTensor4dDescriptor(
		    x_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		    static_cast<int>( input_activations.shape()[0] ),
		    static_cast<int>( input_activations.shape()[1] ),
		    static_cast<int>( input_activations.shape()[2] ),
		    static_cast<int>( input_activations.shape()[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (x) failed" );
		}

		detail::CudnnFilterDesc w_desc;
		st = cudnnSetFilter4dDescriptor( w_desc.desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
		                                 static_cast<int>( ks[0] ), static_cast<int>( ks[1] ),
		                                 static_cast<int>( ks[2] ), static_cast<int>( ks[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetFilter4dDescriptor (w) failed" );
		}

		detail::CudnnFilterDesc dw_desc;
		st = cudnnSetFilter4dDescriptor( dw_desc.desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
		                                 static_cast<int>( ks[0] ), static_cast<int>( ks[1] ),
		                                 static_cast<int>( ks[2] ), static_cast<int>( ks[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetFilter4dDescriptor (dw) failed" );
		}

		detail::CudnnTensorDesc dy_desc;
		st = cudnnSetTensor4dDescriptor(
		    dy_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		    static_cast<int>( grad_wrt_output_nchw.shape()[0] ),
		    static_cast<int>( grad_wrt_output_nchw.shape()[1] ),
		    static_cast<int>( grad_wrt_output_nchw.shape()[2] ),
		    static_cast<int>( grad_wrt_output_nchw.shape()[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (dy) failed" );
		}

		detail::CudnnTensorDesc dx_desc;
		st = cudnnSetTensor4dDescriptor(
		    dx_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		    static_cast<int>( input_activations.shape()[0] ),
		    static_cast<int>( input_activations.shape()[1] ),
		    static_cast<int>( input_activations.shape()[2] ),
		    static_cast<int>( input_activations.shape()[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (dx) failed" );
		}

		detail::CudnnTensorDesc db_desc;
		st = cudnnSetTensor4dDescriptor( db_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
		                                 static_cast<int>( ks[0] ), 1, 1 );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (db) failed" );
		}

		detail::CudnnConvDesc conv_desc;
		st = cudnnSetConvolution2dDescriptor( conv_desc.desc, 1, 1, 1, 1, 1, 1,
		                                        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetConvolution2dDescriptor failed" );
		}

		cudnnConvolutionBwdDataAlgo_t const data_algo    = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
		cudnnConvolutionBwdFilterAlgo_t const filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

		std::size_t ws_data   = 0;
		std::size_t ws_filter = 0;
		st = cudnnGetConvolutionBackwardDataWorkspaceSize(
		    handle, w_desc.desc, dy_desc.desc, conv_desc.desc, dx_desc.desc, data_algo, &ws_data );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnGetConvolutionBackwardDataWorkspaceSize failed" );
		}
		st = cudnnGetConvolutionBackwardFilterWorkspaceSize( handle, x_desc.desc, dy_desc.desc,
		                                                     conv_desc.desc, dw_desc.desc, filter_algo,
		                                                     &ws_filter );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnGetConvolutionBackwardFilterWorkspaceSize failed" );
		}

		std::size_t const ws_size = std::max( ws_data, ws_filter );
		void *workspace = nullptr;
		bool const use_persistent_ws =
		    cudnn_workspace_cache != nullptr && cudnn_workspace_capacity_bytes != nullptr;
		if ( ws_size > 0 ) {
			if ( use_persistent_ws ) {
				cudaError_t const werr = detail::cudnn_workspace_ensure(
				    cudnn_workspace_cache, cudnn_workspace_capacity_bytes, ws_size );
				if ( werr != cudaSuccess ) {
					throw std::runtime_error(
					    "convolutional_backward_cudnn: persistent workspace allocation failed" );
				}
				workspace = *cudnn_workspace_cache;
			} else if ( detail::cuda_malloc_retry( &workspace, ws_size ) != cudaSuccess ) {
				throw std::runtime_error( "convolutional_backward_cudnn: workspace allocation failed" );
			}
		}

		float alpha = 1.f;
		float beta  = 0.f;
		st          = cudnnConvolutionBackwardFilter(
		    handle, &alpha, x_desc.desc, input_activations.data(), dy_desc.desc,
		    grad_wrt_output_nchw.data(), conv_desc.desc, filter_algo, workspace, ws_size, &beta,
		    dw_desc.desc, grad_weights.data() );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			if ( !use_persistent_ws && workspace != nullptr ) {
				(void)cudaFree( workspace );
			}
			throw std::runtime_error( "cudnnConvolutionBackwardFilter failed" );
		}

		st = cudnnConvolutionBackwardData( handle, &alpha, w_desc.desc, weights.data(), dy_desc.desc,
		                                   grad_wrt_output_nchw.data(), conv_desc.desc, data_algo, workspace,
		                                   ws_size, &beta, dx_desc.desc, grad_wrt_input_nchw.data() );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			if ( !use_persistent_ws && workspace != nullptr ) {
				(void)cudaFree( workspace );
			}
			throw std::runtime_error( "cudnnConvolutionBackwardData failed" );
		}

		if ( !use_persistent_ws && workspace != nullptr ) {
			(void)cudaFree( workspace );
		}

		st = cudnnConvolutionBackwardBias( handle, &alpha, dy_desc.desc, grad_wrt_output_nchw.data(), &beta,
		                                   db_desc.desc, grad_bias_rank4.data() );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnConvolutionBackwardBias failed" );
		}
	}
}

/// cuDNN max-pool forward (NCHW), valid pooling (pad 0). Matches \c max_pool_forward_nchw geometry.
template <typename T>
void max_pool_forward_cudnn( CudaTensor4<T> const &input, CudaTensor4<T> &output,
                             CudaTensorNBase<4, std::size_t> &argmax_indices, std::size_t pool_size,
                             std::size_t stride )
{
	(void)argmax_indices;
	if constexpr ( !std::is_same_v<T, float> ) {
		throw std::runtime_error( "max_pool_forward_cudnn: only float is supported" );
	} else {
		cudnnHandle_t &handle = CudnnHandle::instance();
		detail::CudnnTensorDesc input_desc;
		cudnnStatus_t st = cudnnSetTensor4dDescriptor(
		    input_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		    static_cast<int>( input.shape()[0] ),
		    static_cast<int>( input.shape()[1] ),
		    static_cast<int>( input.shape()[2] ),
		    static_cast<int>( input.shape()[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (input) failed" );
		}

		detail::CudnnPoolDesc pooling_desc;
		int const ph = static_cast<int>( pool_size );
		int const st_r = static_cast<int>( stride );
		st = cudnnSetPooling2dDescriptor( pooling_desc.desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, ph, ph, 0, 0,
		                                  st_r, st_r );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetPooling2dDescriptor failed" );
		}

		detail::CudnnTensorDesc output_desc;
		st = cudnnSetTensor4dDescriptor( output_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		                                 static_cast<int>( output.shape()[0] ),
		                                 static_cast<int>( output.shape()[1] ),
		                                 static_cast<int>( output.shape()[2] ),
		                                 static_cast<int>( output.shape()[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (output) failed" );
		}

		float alpha = 1.f;
		float beta  = 0.f;
		st          = cudnnPoolingForward( handle, pooling_desc.desc, &alpha, input_desc.desc, input.data(),
		                                  &beta, output_desc.desc, output.data() );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnPoolingForward failed" );
		}
	}
}

/// cuDNN max-pool backward (NCHW). \p input and \p pooled_output must be the same tensors as in forward;
/// \p argmax_indices is unused (indices are internal to cuDNN).
template <typename T>
void max_pool_backward_cudnn( CudaTensor4<T> const &input, CudaTensor4<T> const &pooled_output,
                              CudaTensor4<T> const &grad_wrt_pooled, CudaTensor4<T> &grad_wrt_input,
                              CudaTensorNBase<4, std::size_t> const &argmax_indices, std::size_t pool_size,
                              std::size_t stride )
{
	(void)argmax_indices;
	if constexpr ( !std::is_same_v<T, float> ) {
		throw std::runtime_error( "max_pool_backward_cudnn: only float is supported" );
	} else {
		cudnnHandle_t &handle = CudnnHandle::instance();

		detail::CudnnTensorDesc x_desc;
		cudnnStatus_t st = cudnnSetTensor4dDescriptor(
		    x_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		    static_cast<int>( input.shape()[0] ),
		    static_cast<int>( input.shape()[1] ),
		    static_cast<int>( input.shape()[2] ),
		    static_cast<int>( input.shape()[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (x) failed" );
		}

		detail::CudnnTensorDesc y_desc;
		st = cudnnSetTensor4dDescriptor(
		    y_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		    static_cast<int>( pooled_output.shape()[0] ),
		    static_cast<int>( pooled_output.shape()[1] ),
		    static_cast<int>( pooled_output.shape()[2] ),
		    static_cast<int>( pooled_output.shape()[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (y) failed" );
		}

		detail::CudnnTensorDesc dy_desc;
		st = cudnnSetTensor4dDescriptor(
		    dy_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		    static_cast<int>( grad_wrt_pooled.shape()[0] ),
		    static_cast<int>( grad_wrt_pooled.shape()[1] ),
		    static_cast<int>( grad_wrt_pooled.shape()[2] ),
		    static_cast<int>( grad_wrt_pooled.shape()[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (dy) failed" );
		}

		detail::CudnnTensorDesc dx_desc;
		st = cudnnSetTensor4dDescriptor(
		    dx_desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		    static_cast<int>( grad_wrt_input.shape()[0] ),
		    static_cast<int>( grad_wrt_input.shape()[1] ),
		    static_cast<int>( grad_wrt_input.shape()[2] ),
		    static_cast<int>( grad_wrt_input.shape()[3] ) );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetTensor4dDescriptor (dx) failed" );
		}

		detail::CudnnPoolDesc pooling_desc;
		int const ph = static_cast<int>( pool_size );
		int const st_r = static_cast<int>( stride );
		st = cudnnSetPooling2dDescriptor( pooling_desc.desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, ph, ph, 0,
		                                0, st_r, st_r );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnSetPooling2dDescriptor failed" );
		}

		float alpha = 1.f;
		float beta  = 0.f;
		st          = cudnnPoolingBackward( handle, pooling_desc.desc, &alpha, y_desc.desc, pooled_output.data(),
		                                   dy_desc.desc, grad_wrt_pooled.data(), x_desc.desc, input.data(), &beta,
		                                   dx_desc.desc, grad_wrt_input.data() );
		if ( st != CUDNN_STATUS_SUCCESS ) {
			throw std::runtime_error( "cudnnPoolingBackward failed" );
		}
	}
}

} // namespace neural

#else // NEURAL_CUDNN_ENABLED -------------------------------------------------

// When cuDNN is not available, fall back to the CPU TensorN so that
// the same template parameters compile in both configurations.
#include "tensor_n.hpp"

namespace neural {

template <std::size_t rank, typename T = float>
using CudaTensorN = TensorN<rank, T>;

} // namespace neural

#endif // NEURAL_CUDNN_ENABLED

#endif // NEURAL_IMPL_CUDA_TENSOR_N_HPP
