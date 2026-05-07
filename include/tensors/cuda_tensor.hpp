#ifndef NEURAL_IMPL_CUDA_TENSOR_HPP
#define NEURAL_IMPL_CUDA_TENSOR_HPP

#include "neural_cuda_kernels.hpp"
#include "neural_cuda_layer_sync.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#if NEURAL_CUDA_ENABLED
#include <cuda_runtime.h>
#include <driver_types.h>
#include "cuda_mem.hpp"
#endif
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <random>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#if NEURAL_CUDA_ENABLED
#include "cublas_handle.hpp"
#include <cublas_v2.h>
#endif
#include "tensor.hpp"

namespace neural {

#if NEURAL_CUDA_ENABLED
template <typename T = float>
class CudaTensor
{
#if NEURAL_CUDA_ENABLED
		using fp8 = __nv_fp8_e4m3;
#endif
	public:
		using value_type = T;
		using size_type = std::size_t;

		template <typename type>
		using tensor_alias = CudaTensor<type>;

	public:
		CudaTensor() noexcept = default;
		CudaTensor( std::size_t rows, std::size_t cols );
		explicit CudaTensor( std::array<std::size_t, 2> shape )
			: CudaTensor( shape[0], shape[1] )
		{
		}

		template <std::random_access_iterator It>
		CudaTensor( std::size_t rows, std::size_t cols, It begin, It end );

		CudaTensor( std::size_t rows, std::size_t cols, value_type value );
		CudaTensor( CudaTensor const &other );
		CudaTensor( CudaTensor &&other ) noexcept;
		CudaTensor &operator=( CudaTensor const &other );
		CudaTensor &operator=( CudaTensor &&other );

		/// Wrap an existing device pointer; if \p own is \c false, \c ~CudaTensor will not \c cudaFree it.
		CudaTensor( T *device_ptr, std::size_t rows, std::size_t cols, bool own ) noexcept;

		~CudaTensor();

		std::size_t rows() const;
		std::size_t cols() const;
		std::size_t size() const;
		std::array<std::size_t, 2> shape() const { return { m_rows, m_cols }; }

		/// Device pointer; do not dereference on the host.
		value_type       *data() noexcept       { return m_data_handle; }
		value_type const *data() const noexcept { return m_data_handle; }

		value_type operator()( std::size_t row, std::size_t col ) const;

		void assign( std::size_t row, std::size_t col, value_type value );
		void assign( value_type value );
		/// Copies \p size elements from \p src (host or device; uses \c cudaMemcpyDefault).
		void assign( value_type const *src, std::size_t size );

		void assignTensorAsRow( std::size_t row, CudaTensor const &other );
		void assignTensor( value_type *value, std::size_t size );

		CudaTensor transpose() const;
		CudaTensor &transposeInPlace();
		CudaTensor reshape( std::size_t rows, std::size_t cols ) const;
		CudaTensor &reshapeInPlace( std::size_t rows, std::size_t cols );

		CudaTensor matmul( CudaTensor const &other ) const;
		CudaTensor &matmulInPlace( CudaTensor const &other );

		CudaTensor &matmulInPlace( CudaTensor const &m1, CudaTensor const &m2,
		                           bool transpose_first = false, bool transpose_second = false );

		CudaTensor &elementwiseMultiplyInPlace( CudaTensor const &other );
		/// Elementwise product of \c *this and \p other into \p out; replaces \p out if shape differs.
		void elementwiseMultiply( CudaTensor const &other, CudaTensor &out ) const;
		CudaTensor &mulNSubstractInPlace( CudaTensor const &other, value_type scalar );
		/// BLAS-style \c axpy: \c *this := alpha * x + *this (float only).
		CudaTensor &axpy( CudaTensor const &x, value_type alpha );
		/// BLAS-style \c geam (in-place): \c *this := alpha * *this + beta * B (float only).
		CudaTensor &geam( value_type alpha, value_type beta, CudaTensor const &B );

		CudaTensor divideRowsWithCol( CudaTensor const &other ) const;
		CudaTensor &divideRowsWithColInPlace( CudaTensor const &other );

		value_type maxCoeff() const;

		CudaTensor operator+( CudaTensor const &other ) const;
		CudaTensor &operator+=( CudaTensor const &other );
		CudaTensor operator-( CudaTensor const &other ) const;
		CudaTensor &operator-=( CudaTensor const &other );
		CudaTensor addColwise( CudaTensor const &col ) const;
		CudaTensor &addColwiseInPlace( CudaTensor const &col );
		CudaTensor subtractColwise( CudaTensor const &col ) const;
		CudaTensor operator*( CudaTensor const &other ) const;
		CudaTensor &operator*=( CudaTensor const &other );
		CudaTensor operator*( value_type scalar ) const;
		CudaTensor &operator*=( value_type scalar );

		CudaTensor cwiseGreater( value_type scalar ) const;
		CudaTensor &cwiseGreaterInPlace( CudaTensor const &other, value_type scalar );
		CudaTensor cwiseOneMinus() const;
		void cwiseSigmoid( CudaTensor &out ) const;
		CudaTensor cwiseSigmoid() const;
		CudaTensor cwiseExp() const;
		CudaTensor cwiseLog() const;

		value_type sum() const;
		value_type asum() const;
		CudaTensor sumAlongAxis( std::size_t axis, bool transpose_out = false ) const;
		CudaTensor sum_along_axis( std::size_t axis ) const { return sumAlongAxis( axis, false ); }

		CudaTensor &sumAlongAxisInPlace( CudaTensor const &src, std::size_t axis,
		                                 bool transpose_out = false );

		CudaTensor max_along_axis( std::size_t axis ) const;

		CudaTensor<std::uint32_t> argmaxAlongAxis( std::size_t axis ) const;

		template <typename Generator>
		void randomize( Generator &generator ) noexcept;
		/// Seeded uniform \f$[0, 1)\f$; uses the \c cuda_random_uniform_float fast path for
		/// \c float. Deterministic given \p seed.
		void randomize( std::uint64_t seed ) noexcept;
		void randomizeHe( std::size_t fan_in, std::uint64_t seed ) noexcept;
		/// Matches PyTorch \c nn.Linear / \c nn.Conv2d \c reset_parameters (see \c Tensor::randomizePytorchDefault).
		void randomizePytorchDefault( std::size_t fan_in, std::uint64_t seed ) noexcept;

		void copyToHost( T *host ) const;
	private:
		/// Non-owning tensors borrow device storage; \c cudaFree / realloc must not run on them.
		void throw_if_non_owning_realloc( char const *context ) const
		{
			if ( !m_own ) {
				throw std::invalid_argument( context );
			}
		}

		std::size_t m_rows = 0;
		std::size_t m_cols = 0;
		T *m_data_handle = nullptr;
		bool m_own = true;
};

template <typename U>
struct is_cuda_tensor<CudaTensor<U>> : std::true_type
{
};

template <typename T>
CudaTensor<T>::CudaTensor( std::size_t rows, std::size_t cols )
    : m_rows( rows )
    , m_cols( cols )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( rows == 0 || cols == 0 ) {
		return;
	}
	cudaError_t cuda_stat = detail::cuda_malloc_retry(
	    reinterpret_cast<void **>( &m_data_handle ), rows * cols * sizeof( T ) );
	if ( cuda_stat != cudaSuccess ) {
		throw std::runtime_error( "device memory allocation failed" );
	}
	if constexpr ( std::is_same_v<T, float> ) {
		cuda_stat = cudaMemset( m_data_handle, 0, rows * cols * sizeof( T ) );
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cuda_stat = cuda_fill_fp8( m_data_handle, rows * cols, fp8{} );
#endif
	} else {
		std::vector<T> host( rows * cols, T{} );
		cuda_stat = cudaMemcpy( m_data_handle, host.data(), rows * cols * sizeof( T ),
		                        cudaMemcpyHostToDevice );
	}
	if ( cuda_stat != cudaSuccess ) {
		cudaFree( m_data_handle );
		m_data_handle = nullptr;
		throw std::runtime_error( "CudaTensor(rows,cols): device initialization failed" );
	}
}

template <typename T>
template <std::random_access_iterator It>
CudaTensor<T>::CudaTensor( std::size_t rows, std::size_t cols, It begin, It end )
    : m_rows( rows )
    , m_cols( cols )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	std::size_t const n = rows * cols;
	std::ptrdiff_t const dist = std::distance( begin, end );
	if ( dist < 0 || static_cast<std::size_t>( dist ) != n ) {
		throw std::invalid_argument(
		    "CudaTensor(rows,cols,begin,end): iterator range size does not match rows*cols" );
	}
	if ( n == 0 ) {
		return;
	}
	cudaError_t cuda_stat =
	    detail::cuda_malloc_retry( reinterpret_cast<void **>( &m_data_handle ), n * sizeof( T ) );
	if ( cuda_stat != cudaSuccess ) {
		throw std::runtime_error( "device memory allocation failed" );
	}
	std::vector<T> host( begin, end );
	cuda_stat = cudaMemcpy( m_data_handle, host.data(), n * sizeof( T ), cudaMemcpyHostToDevice );
	if ( cuda_stat != cudaSuccess ) {
		cudaFree( m_data_handle );
		m_data_handle = nullptr;
		throw std::runtime_error( "CudaTensor iterator ctor: host to device copy failed" );
	}
}

template <typename T>
CudaTensor<T>::~CudaTensor()
{
	if ( m_own && m_data_handle != nullptr ) {
		::neural::cuda_layer_sync();
		cudaFree( m_data_handle );
		m_data_handle = nullptr;
	}
}

template <typename T>
CudaTensor<T>::CudaTensor( std::size_t rows, std::size_t cols, value_type value )
    : m_rows( rows )
    , m_cols( cols )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	cudaError_t cuda_stat = detail::cuda_malloc_retry(
	    reinterpret_cast<void **>( &m_data_handle ), rows * cols * sizeof( T ) );
	if ( cuda_stat != cudaSuccess ) {
		throw std::runtime_error( "device memory allocation failed" );
	}
	if constexpr ( std::is_same_v<T, float> ) {
		cuda_stat = cuda_fill_float( m_data_handle, rows * cols, value );
	}
#if NEURAL_CUDA_ENABLED
	else if constexpr ( std::is_same_v<T, fp8> ) {
		cuda_stat = cuda_fill_fp8( m_data_handle, rows * cols, value );
	}
#endif
	else {
		std::vector<T> host( rows * cols, value );
		cuda_stat = cudaMemcpy( m_data_handle, host.data(), rows * cols * sizeof( T ),
		                        cudaMemcpyHostToDevice );
	}
	if ( cuda_stat != cudaSuccess ) {
		cudaFree( m_data_handle );
		m_data_handle = nullptr;
		throw std::runtime_error( "device fill failed" );
	}
}

template <typename T>
CudaTensor<T>::CudaTensor( CudaTensor const &other )
    : m_rows( other.m_rows )
    , m_cols( other.m_cols )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( other.m_data_handle != nullptr ) {
		cudaError_t cuda_stat = detail::cuda_malloc_retry(
		    reinterpret_cast<void **>( &m_data_handle ),
		    other.m_rows * other.m_cols * sizeof( T ) );
		if ( cuda_stat != cudaSuccess ) {
			throw std::runtime_error( "device memory allocation failed" );
		}
		cuda_stat = cudaMemcpy( m_data_handle, other.m_data_handle,
		                        other.m_rows * other.m_cols * sizeof( T ),
		                        cudaMemcpyDeviceToDevice );
		if ( cuda_stat != cudaSuccess ) {
			cudaFree( m_data_handle );
			m_data_handle = nullptr;
			throw std::runtime_error( "device memory copy failed" );
		}
	} else {
		m_data_handle = nullptr;
	}
}

template <typename T>
CudaTensor<T>::CudaTensor( T *device_ptr, std::size_t rows, std::size_t cols, bool own ) noexcept
    : m_rows( rows )
    , m_cols( cols )
    , m_data_handle( device_ptr )
    , m_own( own )
{
}

template <typename T>
CudaTensor<T>::CudaTensor( CudaTensor &&other ) noexcept
    : m_rows( other.m_rows )
    , m_cols( other.m_cols )
    , m_data_handle( other.m_data_handle )
    , m_own( other.m_own )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	other.m_data_handle = nullptr;
	other.m_rows = 0;
	other.m_cols = 0;
	other.m_own = true;
}

template <typename T>
CudaTensor<T> &CudaTensor<T>::operator=( CudaTensor const &other )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( this == &other ) {
		return *this;
	}
	if ( !m_own ) {
		if ( m_rows != other.m_rows || m_cols != other.m_cols ) {
			throw std::invalid_argument(
			    "CudaTensor::operator=(CudaTensor const &): non-owning tensor requires matching shape" );
		}
		if ( other.m_data_handle == nullptr || size() == 0 ) {
			return *this;
		}
		cudaError_t const cuda_stat =
		    cudaMemcpy( m_data_handle, other.m_data_handle, size() * sizeof( T ),
		                cudaMemcpyDeviceToDevice );
		if ( cuda_stat != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::operator=(const &): device memory copy failed" );
		}
		return *this;
	}

	if ( other.m_data_handle == nullptr ) {
		if ( m_data_handle != nullptr ) {
			cudaFree( m_data_handle );
			m_data_handle = nullptr;
		}
		m_rows = other.m_rows;
		m_cols = other.m_cols;
		return *this;
	}
	if ( m_data_handle != nullptr
	     && m_rows * m_cols == other.m_rows * other.m_cols ) {
		m_rows = other.m_rows;
		m_cols = other.m_cols;
		cudaError_t const cuda_stat =
		    cudaMemcpy( m_data_handle, other.m_data_handle,
		                other.m_rows * other.m_cols * sizeof( T ),
		                cudaMemcpyDeviceToDevice );
		if ( cuda_stat != cudaSuccess ) {
			throw std::runtime_error( "device memory copy failed" );
		}
		return *this;
	}
	m_rows = other.m_rows;
	m_cols = other.m_cols;
	if ( m_data_handle != nullptr ) {
		cudaFree( m_data_handle );
		m_data_handle = nullptr;
	}
	cudaError_t cuda_stat = detail::cuda_malloc_retry(
	    reinterpret_cast<void **>( &m_data_handle ),
	    other.m_rows * other.m_cols * sizeof( T ) );
	if ( cuda_stat != cudaSuccess ) {
		throw std::runtime_error( "device memory allocation failed" );
	}
	cuda_stat = cudaMemcpy( m_data_handle, other.m_data_handle,
	                        other.m_rows * other.m_cols * sizeof( T ),
	                        cudaMemcpyDeviceToDevice );
	if ( cuda_stat != cudaSuccess ) {
		cudaFree( m_data_handle );
		m_data_handle = nullptr;
		throw std::runtime_error( "device memory copy failed" );
	}
	return *this;
}

template <typename T>
CudaTensor<T> &CudaTensor<T>::operator=( CudaTensor &&other )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( this == &other ) {
		return *this;
	}
	throw_if_non_owning_realloc(
	    "CudaTensor::operator=(CudaTensor &&): cannot move-assign into a non-owning tensor" );
	if ( m_own && m_data_handle != nullptr ) {
		cudaFree( m_data_handle );
	}
	m_rows = other.m_rows;
	m_cols = other.m_cols;
	m_data_handle = other.m_data_handle;
	m_own = other.m_own;
	other.m_data_handle = nullptr;
	other.m_rows = 0;
	other.m_cols = 0;
	other.m_own = true;
	return *this;
}

template <typename T>
std::size_t CudaTensor<T>::rows() const
{
	return m_rows;
}

template <typename T>
std::size_t CudaTensor<T>::cols() const
{
	return m_cols;
}

template <typename T>
std::size_t CudaTensor<T>::size() const
{
	return m_rows * m_cols;
}

template <typename T>
typename CudaTensor<T>::value_type CudaTensor<T>::operator()( std::size_t row,
                                                              std::size_t col ) const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( row >= m_rows || col >= m_cols ) {
		throw std::out_of_range( "CudaTensor::operator(): index out of range" );
	}
	if ( m_data_handle == nullptr || m_rows == 0 || m_cols == 0 ) {
		return value_type{};
	}
	T value{};
	cudaError_t const cuda_stat =
	    cudaMemcpy( &value, m_data_handle + row * m_cols + col, sizeof( T ),
	                cudaMemcpyDeviceToHost );
	if ( cuda_stat != cudaSuccess ) {
		throw std::runtime_error( "CudaTensor::operator(): device to host copy failed" );
	}

	return value;
}

template <typename T>
void CudaTensor<T>::assign( std::size_t row, std::size_t col, value_type value )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( row >= m_rows || col >= m_cols ) {
		throw std::out_of_range( "CudaTensor::assign(): index out of range" );
	}
	if ( m_data_handle == nullptr || m_rows == 0 || m_cols == 0 ) {
		return;
	}
	cudaError_t const cuda_stat =
	    cudaMemcpy( m_data_handle + row * m_cols + col, &value, sizeof( T ), cudaMemcpyHostToDevice );
	
	if ( cuda_stat != cudaSuccess ) {
		throw std::runtime_error( "CudaTensor::assign(): host to device copy failed" );
	}
}

template <typename T>
void CudaTensor<T>::assign( value_type value )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_data_handle == nullptr || size() == 0 ) {
		return;
	}
	cudaError_t cuda_stat = cudaSuccess;
	if constexpr ( std::is_same_v<T, float> ) {
		cuda_stat = cuda_fill_float( m_data_handle, size(), value );
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cuda_stat = cuda_fill_fp8( m_data_handle, size(), value );
#endif
	} else {
		std::vector<T> host( size(), value );
		cuda_stat = cudaMemcpy( m_data_handle, host.data(), size() * sizeof( T ),
		                        cudaMemcpyHostToDevice );
	}
	if ( cuda_stat != cudaSuccess ) {
		throw std::runtime_error( "CudaTensor::assign(): device fill failed" );
	}
}

template <typename T>
void CudaTensor<T>::assignTensorAsRow( std::size_t row, CudaTensor const &other )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( row >= m_rows ) {
		throw std::out_of_range( "CudaTensor::assignTensorAsRow(): row out of range" );
	}
	if ( m_data_handle == nullptr || m_cols == 0 ) {
		return;
	}
	if ( other.m_data_handle == nullptr || other.m_rows < 1 ) {
		throw std::invalid_argument(
		    "CudaTensor::assignTensorAsRow(): other must have rows>=1" );
	}
	if ( other.m_cols != m_cols && other.m_rows * other.m_cols != m_cols ) {
		throw std::invalid_argument(
		    "CudaTensor::assignTensorAsRow(): other.cols() or other.size() must match destination cols" );
	}
	cudaError_t const cuda_stat =
	    cudaMemcpy( m_data_handle + row * m_cols, other.m_data_handle,
	                m_cols * sizeof( T ), cudaMemcpyDeviceToDevice );
	if ( cuda_stat != cudaSuccess ) {
		throw std::runtime_error( "CudaTensor::assignTensorAsRow(): cudaMemcpy failed" );
	}
}

template <typename T>
void CudaTensor<T>::assignTensor( value_type *value, std::size_t size )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( size != m_rows * m_cols ) {
		throw std::out_of_range( "CudaTensor::assignTensorAsRow(): row out of range" );
	}

	cudaError_t const cuda_stat =
	    cudaMemcpy( m_data_handle, value, size * sizeof( T ),
	                cudaMemcpyHostToDevice );
	if ( cuda_stat != cudaSuccess ) {
		throw std::runtime_error( "CudaTensor::assignTensor(): cudaMemcpy failed" );
	}
}

template <typename T>
void CudaTensor<T>::assign( value_type const *src, std::size_t size )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( size != m_rows * m_cols ) {
		throw std::out_of_range( "CudaTensor::assign(ptr): size mismatch" );
	}
	cudaError_t const cuda_stat =
	    cudaMemcpy( m_data_handle, src, size * sizeof( T ), cudaMemcpyDefault );
	if ( cuda_stat != cudaSuccess ) {
		throw std::runtime_error( "CudaTensor::assign(ptr): cudaMemcpy failed" );
	}
}

template <typename T>
CudaTensor<T> CudaTensor<T>::transpose() const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_data_handle == nullptr || m_rows == 0 || m_cols == 0 ) {
		return CudaTensor<T>( m_cols, m_rows );
	}
	CudaTensor<T> out( m_cols, m_rows );
	if constexpr ( std::is_same_v<T, float> ) {
		cudaError_t const err =
		    cuda_transpose_float( m_data_handle, out.m_data_handle, m_rows, m_cols );
		if ( err != cudaSuccess ) {
			return CudaTensor<T>{};
		}
		return out;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cudaError_t const err = cuda_transpose_fp8( m_data_handle, out.m_data_handle, m_rows, m_cols );
		if ( err != cudaSuccess ) {
			return CudaTensor<T>{};
		}
		return out;
#endif
	} else {
		(void)out;
		return CudaTensor<T>{};
	}
}

template <typename T>
CudaTensor<T> &CudaTensor<T>::transposeInPlace()
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	throw_if_non_owning_realloc(
	    "CudaTensor::transposeInPlace(): would replace storage on a non-owning tensor" );
	*this = transpose();
	return *this;
}

template <typename T>
CudaTensor<T> CudaTensor<T>::reshape( std::size_t rows, std::size_t cols ) const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	CudaTensor<T> out( *this );
	out.reshapeInPlace( rows, cols );

	return out;
}

template <typename T>
CudaTensor<T> &CudaTensor<T>::reshapeInPlace( std::size_t rows, std::size_t cols )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( rows * cols != m_rows * m_cols ) {
		throw std::invalid_argument(
		    "CudaTensor::reshapeInPlace: new shape must have the same number of elements" );
	}
	m_rows = rows;
	m_cols = cols;

	return *this;
}

template <typename T>
CudaTensor<T> CudaTensor<T>::matmul( CudaTensor const &other ) const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_data_handle == nullptr || other.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::matmul(): one of the tensors is not initialized" );
	}
	if ( m_cols != other.m_rows ) {
		throw std::invalid_argument( "CudaTensor::matmul(): incompatible dimensions" );
	}
	CudaTensor<T> out( m_rows, other.m_cols );
	if constexpr ( std::is_same_v<T, float> ) {
		float const alpha = 1.0f;
		float const beta = 0.0f;
		cublasStatus_t const st = cublasSgemm(
		    CublasHandle::instance(),
		    CUBLAS_OP_N,
		    CUBLAS_OP_N,
		    static_cast<int>( other.m_cols ),
		    static_cast<int>( m_rows ),
		    static_cast<int>( m_cols ),
		    &alpha,
		    other.m_data_handle,
		    static_cast<int>( other.m_cols ),
		    m_data_handle,
		    static_cast<int>( m_cols ),
		    &beta,
		    out.m_data_handle,
		    static_cast<int>( other.m_cols ) );
		if ( st != CUBLAS_STATUS_SUCCESS ) {
			return CudaTensor<T>{};
		}
		return out;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cudaError_t const cuda_stat = cuda_matmul_fp8( m_data_handle, other.m_data_handle, out.m_data_handle, m_rows, m_cols, other.m_rows, other.m_cols );
		if ( cuda_stat != cudaSuccess ) {
			return CudaTensor<T>{};
		}
		return out;
#endif
	} else {
		(void)out;
		return CudaTensor<T>{};
	}
}

template <typename T>
CudaTensor<T> &CudaTensor<T>::matmulInPlace( CudaTensor const &other )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	throw_if_non_owning_realloc(
	    "CudaTensor::matmulInPlace(): would replace storage on a non-owning tensor" );
	CudaTensor<T> out = matmul( other );
	*this = std::move( out );
	return *this;
}

template <typename T>
CudaTensor<T> &CudaTensor<T>::matmulInPlace( CudaTensor const &m1, CudaTensor const &m2,
                                             bool transpose_first, bool transpose_second )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_data_handle == nullptr || m1.m_data_handle == nullptr || m2.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::matmulInPlace(m1,m2): one of the tensors is not initialized" );
	}
	auto const throw_dim = [] {
		throw std::invalid_argument(
		    "CudaTensor::matmulInPlace(m1,m2,transpose_first,transpose_second): incompatible dimensions" );
	};
	auto const fp8_transpose_throw = [] {
		throw std::runtime_error(
		    "CudaTensor::matmulInPlace(m1,m2): fp8 with transpose is not supported" );
	};

	if constexpr ( std::is_same_v<T, float> ) {
		float const alpha = 1.0f;
		float const beta = 0.0f;
		cublasOperation_t const op_b = transpose_second ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t const op_a = transpose_first ? CUBLAS_OP_T : CUBLAS_OP_N;

		if ( !transpose_first && !transpose_second ) {
			if ( m1.m_cols != m2.m_rows || m1.m_rows != m_rows || m2.m_cols != m_cols ) {
				throw_dim();
			}
			cublasStatus_t const st = cublasSgemm(
			    CublasHandle::instance(),
			    op_b,
			    op_a,
			    static_cast<int>( m2.m_cols ),
			    static_cast<int>( m_rows ),
			    static_cast<int>( m1.m_cols ),
			    &alpha,
			    m2.m_data_handle,
			    static_cast<int>( m2.m_cols ),
			    m1.m_data_handle,
			    static_cast<int>( m1.m_cols ),
			    &beta,
			    this->m_data_handle,
			    static_cast<int>( m2.m_cols ) );
			if ( st != CUBLAS_STATUS_SUCCESS ) {
				throw std::runtime_error( "CudaTensor::matmulInPlace(m1,m2): cublasSgemm failed" );
			}
			return *this;
		}
		if ( transpose_first && !transpose_second ) {
			if ( m1.m_rows != m2.m_rows || m1.m_cols != m_rows || m2.m_cols != m_cols ) {
				throw_dim();
			}
			cublasStatus_t const st = cublasSgemm(
			    CublasHandle::instance(),
			    op_b,
			    op_a,
			    static_cast<int>( m2.m_cols ),
			    static_cast<int>( m1.m_cols ),
			    static_cast<int>( m1.m_rows ),
			    &alpha,
			    m2.m_data_handle,
			    static_cast<int>( m2.m_cols ),
			    m1.m_data_handle,
			    static_cast<int>( m1.m_cols ),
			    &beta,
			    this->m_data_handle,
			    static_cast<int>( m2.m_cols ) );
			if ( st != CUBLAS_STATUS_SUCCESS ) {
				throw std::runtime_error( "CudaTensor::matmulInPlace(m1,m2): cublasSgemm failed" );
			}
			return *this;
		}
		if ( !transpose_first && transpose_second ) {
			if ( m1.m_cols != m2.m_cols || m1.m_rows != m_rows || m2.m_rows != m_cols ) {
				throw_dim();
			}
			cublasStatus_t const st = cublasSgemm(
			    CublasHandle::instance(),
			    op_b,
			    op_a,
			    static_cast<int>( m2.m_rows ),
			    static_cast<int>( m1.m_rows ),
			    static_cast<int>( m1.m_cols ),
			    &alpha,
			    m2.m_data_handle,
			    static_cast<int>( m2.m_cols ),
			    m1.m_data_handle,
			    static_cast<int>( m1.m_cols ),
			    &beta,
			    this->m_data_handle,
			    static_cast<int>( m2.m_rows ) );
			if ( st != CUBLAS_STATUS_SUCCESS ) {
				throw std::runtime_error( "CudaTensor::matmulInPlace(m1,m2): cublasSgemm failed" );
			}
			return *this;
		}
		if ( m1.m_rows != m2.m_cols || m1.m_cols != m_rows || m2.m_rows != m_cols ) {
			throw_dim();
		}
		cublasStatus_t const st = cublasSgemm(
		    CublasHandle::instance(),
		    op_b,
		    op_a,
		    static_cast<int>( m2.m_rows ),
		    static_cast<int>( m1.m_cols ),
		    static_cast<int>( m1.m_rows ),
		    &alpha,
		    m2.m_data_handle,
		    static_cast<int>( m2.m_cols ),
		    m1.m_data_handle,
		    static_cast<int>( m1.m_cols ),
		    &beta,
		    this->m_data_handle,
		    static_cast<int>( m2.m_rows ) );
		if ( st != CUBLAS_STATUS_SUCCESS ) {
			throw std::runtime_error( "CudaTensor::matmulInPlace(m1,m2): cublasSgemm failed" );
		}
		return *this;
	}
#if NEURAL_CUDA_ENABLED
	else if constexpr ( std::is_same_v<T, fp8> ) {
		if ( transpose_first || transpose_second ) {
			fp8_transpose_throw();
		}
		if ( m1.m_cols != m2.m_rows || m1.m_rows != m_rows || m2.m_cols != m_cols ) {
			throw_dim();
		}
		cudaError_t const cuda_stat = cuda_matmul_fp8( m1.m_data_handle, m2.m_data_handle, this->m_data_handle, m1.m_rows, m1.m_cols, m2.m_rows, m2.m_cols );
		if ( cuda_stat != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::matmulInPlace(m1,m2): cuda_matmul_fp8 failed" );
		}
		return *this;
	}
#endif
	else {
		(void)transpose_first;
		(void)transpose_second;
		(void)m1;
		(void)m2;
		return *this;
	}
}

template <typename T>
CudaTensor<T> &CudaTensor<T>::elementwiseMultiplyInPlace( CudaTensor const &other )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_rows != other.m_rows || m_cols != other.m_cols ) {
		throw std::invalid_argument(
		    "CudaTensor::elementwiseMultiplyInPlace(): incompatible dimensions" );
	}
	if ( size() == 0 ) {
		return *this;
	}
	if ( m_data_handle == nullptr || other.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::elementwiseMultiplyInPlace(): tensor not initialized" );
	}
	if constexpr ( std::is_same_v<T, float> ) {
		cudaError_t const err = cuda_mul_float_inplace( m_data_handle, other.m_data_handle, size() );
		if ( err != cudaSuccess ) {
			throw std::runtime_error(
			    "CudaTensor::elementwiseMultiplyInPlace(): cuda_mul_float_inplace failed" );
		}
		return *this;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cudaError_t const err = cuda_mul_fp8_inplace( m_data_handle, other.m_data_handle, size() );
		if ( err != cudaSuccess ) {
			throw std::runtime_error(
			    "CudaTensor::elementwiseMultiplyInPlace(): cuda_mul_fp8_inplace failed" );
		}
		return *this;
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::elementwiseMultiplyInPlace(): unsupported type" );
	}
}

template <typename T>
void CudaTensor<T>::elementwiseMultiply( CudaTensor const &other, CudaTensor &out ) const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_rows != other.m_rows || m_cols != other.m_cols ) {
		throw std::invalid_argument(
		    "CudaTensor::elementwiseMultiply(): incompatible dimensions" );
	}
	if ( out.m_rows != m_rows || out.m_cols != m_cols ) {
		out.throw_if_non_owning_realloc(
		    "CudaTensor::elementwiseMultiply(): cannot resize a non-owning output tensor" );
		out = CudaTensor<T>( m_rows, m_cols );
	}
	if ( size() == 0 ) {
		return;
	}
	if ( m_data_handle == nullptr || other.m_data_handle == nullptr
	     || out.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::elementwiseMultiply(): tensor not initialized" );
	}
	if constexpr ( std::is_same_v<T, float> ) {
		cudaError_t const err =
		    cuda_mul_float( m_data_handle, other.m_data_handle, out.m_data_handle, size() );
		if ( err != cudaSuccess ) {
			throw std::runtime_error(
			    "CudaTensor::elementwiseMultiply(): cuda_mul_float failed" );
		}
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cudaError_t const err =
		    cuda_mul_fp8( m_data_handle, other.m_data_handle, out.m_data_handle, size() );
		if ( err != cudaSuccess ) {
			throw std::runtime_error(
			    "CudaTensor::elementwiseMultiply(): cuda_mul_fp8 failed" );
		}
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::elementwiseMultiply(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> &CudaTensor<T>::mulNSubstractInPlace( CudaTensor const &other, value_type scalar )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_data_handle == nullptr || other.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::mulNSubstractInPlace(): tensor not initialized" );
	}
	if ( size() != other.size() ) {
		throw std::invalid_argument( "CudaTensor::mulNSubstractInPlace(): incompatible sizes" );
	}
	if ( size() == 0 ) {
		return *this;
	}
	if constexpr ( std::is_same_v<T, float> ) {
		cudaError_t const err =
		    cuda_mul_substract_float( m_data_handle, other.m_data_handle, size(),
		                              static_cast<float>( scalar ) );
		if ( err != cudaSuccess ) {
			throw std::runtime_error(
			    "CudaTensor::mulNSubstractInPlace(): cuda_mul_substract_float failed" );
		}
		return *this;
	} else {
		throw std::invalid_argument( "CudaTensor::mulNSubstractInPlace(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> &CudaTensor<T>::axpy( CudaTensor const &x, value_type alpha )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_rows != x.m_rows || m_cols != x.m_cols ) {
		throw std::invalid_argument( "CudaTensor::axpy(): incompatible dimensions" );
	}
	if ( size() == 0 ) {
		return *this;
	}
	if ( m_data_handle == nullptr || x.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::axpy(): tensor not initialized" );
	}
	if constexpr ( std::is_same_v<T, float> ) {
		float a = static_cast<float>( alpha );
		cublasStatus_t const st = cublasSaxpy( CublasHandle::instance(), static_cast<int>( size() ),
		                                       &a, x.m_data_handle, 1, m_data_handle, 1 );
		if ( st != CUBLAS_STATUS_SUCCESS ) {
			throw std::runtime_error( "CudaTensor::axpy(): cublasSaxpy failed" );
		}
		return *this;
	} else {
		throw std::invalid_argument( "CudaTensor::axpy(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> &CudaTensor<T>::geam( value_type alpha, value_type beta, CudaTensor const &B )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_rows != B.m_rows || m_cols != B.m_cols ) {
		throw std::invalid_argument( "CudaTensor::geam(): incompatible dimensions" );
	}
	if ( size() == 0 ) {
		return *this;
	}
	if ( m_data_handle == nullptr || B.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::geam(): tensor not initialized" );
	}
	if constexpr ( std::is_same_v<T, float> ) {
		float a = static_cast<float>( alpha );
		float b = static_cast<float>( beta );
		cublasStatus_t const st = cublasSgeam(
		    CublasHandle::instance(),
		    CUBLAS_OP_N,
		    CUBLAS_OP_N,
		    static_cast<int>( m_cols ),
		    static_cast<int>( m_rows ),
		    &a,
		    m_data_handle,
		    static_cast<int>( m_cols ),
		    &b,
		    B.m_data_handle,
		    static_cast<int>( m_cols ),
		    m_data_handle,
		    static_cast<int>( m_cols ) );
		if ( st != CUBLAS_STATUS_SUCCESS ) {
			throw std::runtime_error( "CudaTensor::geam(): cublasSgeam failed" );
		}
		return *this;
	} else {
		throw std::invalid_argument( "CudaTensor::geam(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> CudaTensor<T>::divideRowsWithCol( CudaTensor const &other ) const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_data_handle == nullptr || other.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::divideRowsWithCol(): one of the tensors is not initialized" );
	}
	if ( m_cols != other.m_cols ) {
		throw std::invalid_argument( "CudaTensor::divideRowsWithCol(): incompatible dimensions" );
	}
	CudaTensor<T> out( *this );

	if constexpr ( std::is_same_v<T, float> ) {
		cudaError_t const cuda_stat = divide_rows_with_col_float(
		    out.m_data_handle, other.m_data_handle, m_rows, m_cols );
		if ( cuda_stat != cudaSuccess ) {
			return CudaTensor<T>{};
		}

		return out;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cudaError_t const cuda_stat = divide_rows_with_col_fp8(
		    out.m_data_handle, other.m_data_handle, m_rows, m_cols );
		if ( cuda_stat != cudaSuccess ) {
			return CudaTensor<T>{};
		}

		return out;
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::divideRowsWithCol(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> &CudaTensor<T>::divideRowsWithColInPlace( CudaTensor const &other )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_data_handle == nullptr || other.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::divideRowsWithColInPlace(): one of the tensors is not initialized" );
	}
	if ( m_rows != other.m_rows || other.m_cols != 1u ) {
		throw std::invalid_argument( "CudaTensor::divideRowsWithColInPlace(): incompatible dimensions" );
	}
	cudaError_t cuda_stat = cudaSuccess;
	if constexpr ( std::is_same_v<T, float> ) {
		cuda_stat = divide_rows_with_col_float( m_data_handle, other.m_data_handle, m_rows,
		                                          m_cols );
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cuda_stat = divide_rows_with_col_fp8( m_data_handle, other.m_data_handle, m_rows,
		                                      m_cols );
#endif
	} else {
		throw std::invalid_argument(
		    "CudaTensor::divideRowsWithColInPlace(): unsupported type" );
	}
	if ( cuda_stat != cudaSuccess ) {
		throw std::runtime_error( "CudaTensor::divideRowsWithColInPlace(): device operation failed" );
	}

	return *this;
}

template <typename T>
typename CudaTensor<T>::value_type CudaTensor<T>::maxCoeff() const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_data_handle == nullptr || size() == 0 ) {
		return value_type{};
	}
	if constexpr ( std::is_same_v<T, float> ) {
		float m = 0.f;
		cudaError_t const err = cuda_max_abs_float( m_data_handle, size(), &m );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::maxCoeff(): cuda_max_abs_float failed" );
		}
		return m;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		float m = 0.f;
		cudaError_t const err = cuda_max_abs_fp8( m_data_handle, size(), &m );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::maxCoeff(): cuda_max_abs_fp8 failed" );
		}
		return static_cast<fp8>( m );
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::maxCoeff(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> CudaTensor<T>::operator+( CudaTensor const &other ) const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_rows != other.m_rows || m_cols != other.m_cols ) {
		throw std::invalid_argument( "CudaTensor::operator+(): incompatible dimensions" );
	}
	if ( size() == 0 ) {
		return CudaTensor<T>{};
	}
	if ( m_data_handle == nullptr || other.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::operator+(): one of the tensors is not initialized" );
	}
	CudaTensor<T> out( m_rows, m_cols, value_type{} );
	if constexpr ( std::is_same_v<T, float> ) {
		float const alpha = 1.0f;
		float const beta = 1.0f;
		cublasStatus_t const st = cublasSgeam(
		    CublasHandle::instance(),
		    CUBLAS_OP_N,
		    CUBLAS_OP_N,
		    static_cast<int>( m_cols ),
		    static_cast<int>( m_rows ),
		    &alpha,
		    m_data_handle,
		    static_cast<int>( m_cols ),
		    &beta,
		    other.m_data_handle,
		    static_cast<int>( m_cols ),
		    out.m_data_handle,
		    static_cast<int>( m_cols ) );
		if ( st != CUBLAS_STATUS_SUCCESS ) {
			throw std::runtime_error( "CudaTensor::operator+(): cublasSgeam failed" );
		}
		return out;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cudaError_t const err = cuda_add_fp8(
		    m_data_handle, other.m_data_handle, out.m_data_handle, size(), 1.0f, 1.0f );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::operator+(): cuda_add_fp8 failed" );
		}
		return out;
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::operator+(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> &CudaTensor<T>::operator+=( CudaTensor const &other )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_rows != other.m_rows || m_cols != other.m_cols ) {
		throw std::invalid_argument( "CudaTensor::operator+=(): incompatible dimensions" );
	}
	if ( size() == 0 ) {
		return *this;
	}
	if ( m_data_handle == nullptr || other.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::operator+=(): one of the tensors is not initialized" );
	}
	if constexpr ( std::is_same_v<T, float> ) {
		float const alpha = 1.0f;
		float const beta = 1.0f;
		cublasStatus_t const st = cublasSgeam(
		    CublasHandle::instance(),
		    CUBLAS_OP_N,
		    CUBLAS_OP_N,
		    static_cast<int>( m_cols ),
		    static_cast<int>( m_rows ),
		    &alpha,
		    m_data_handle,
		    static_cast<int>( m_cols ),
		    &beta,
		    other.m_data_handle,
		    static_cast<int>( m_cols ),
		    m_data_handle,
		    static_cast<int>( m_cols ) );
		if ( st != CUBLAS_STATUS_SUCCESS ) {
			throw std::runtime_error( "CudaTensor::operator+=(): cublasSgeam failed" );
		}
		return *this;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cudaError_t const err = cuda_add_fp8(
		    m_data_handle, other.m_data_handle, m_data_handle, size(), 1.0f, 1.0f );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::operator+=(): cuda_add_fp8 failed" );
		}
		return *this;
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::operator+=(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> CudaTensor<T>::operator-( CudaTensor const &other ) const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_rows != other.m_rows || m_cols != other.m_cols ) {
		throw std::invalid_argument( "CudaTensor::operator+(): incompatible dimensions" );
	}
	if ( size() == 0 ) {
		return CudaTensor<T>{};
	}
	if ( m_data_handle == nullptr || other.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::operator+(): one of the tensors is not initialized" );
	}
	CudaTensor<T> out( m_rows, m_cols, value_type{} );
	if constexpr ( std::is_same_v<T, float> ) {
		float const alpha = 1.0f;
		float const beta = -1.0f;
		cublasStatus_t const st = cublasSgeam(
		    CublasHandle::instance(),
		    CUBLAS_OP_N,
		    CUBLAS_OP_N,
		    static_cast<int>( m_cols ),
		    static_cast<int>( m_rows ),
		    &alpha,
		    m_data_handle,
		    static_cast<int>( m_cols ),
		    &beta,
		    other.m_data_handle,
		    static_cast<int>( m_cols ),
		    out.m_data_handle,
		    static_cast<int>( m_cols ) );
		if ( st != CUBLAS_STATUS_SUCCESS ) {
			throw std::runtime_error( "CudaTensor::operator+(): cublasSgeam failed" );
		}
		return out;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cudaError_t const err = cuda_add_fp8(
		    m_data_handle, other.m_data_handle, out.m_data_handle, size(), 1.0f, -1.0f );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::operator+(): cuda_add_fp8 failed" );
		}
		return out;
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::operator+(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> &CudaTensor<T>::operator-=( CudaTensor const &other )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_rows != other.m_rows || m_cols != other.m_cols ) {
		throw std::invalid_argument( "CudaTensor::operator+=(): incompatible dimensions" );
	}
	if ( size() == 0 ) {
		return *this;
	}
	if ( m_data_handle == nullptr || other.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::operator+=(): one of the tensors is not initialized" );
	}
	if constexpr ( std::is_same_v<T, float> ) {
		float const alpha = 1.0f;
		float const beta = -1.0f;
		cublasStatus_t const st = cublasSgeam(
		    CublasHandle::instance(),
		    CUBLAS_OP_N,
		    CUBLAS_OP_N,
		    static_cast<int>( m_cols ),
		    static_cast<int>( m_rows ),
		    &alpha,
		    m_data_handle,
		    static_cast<int>( m_cols ),
		    &beta,
		    other.m_data_handle,
		    static_cast<int>( m_cols ),
		    m_data_handle,
		    static_cast<int>( m_cols ) );
		if ( st != CUBLAS_STATUS_SUCCESS ) {
			throw std::runtime_error( "CudaTensor::operator+=(): cublasSgeam failed" );
		}
		return *this;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cudaError_t const err = cuda_add_fp8(
		    m_data_handle, other.m_data_handle, m_data_handle, size(), 1.0f, -1.0f );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::operator+=(): cuda_add_fp8 failed" );
		}
		return *this;
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::operator+=(): unsupported type" );
	}
	
	return *this;
}

template <typename T>
CudaTensor<T> CudaTensor<T>::addColwise( CudaTensor const &col ) const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_data_handle == nullptr || col.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::addColwise(): one of the tensors is not initialized" );
	}
	if ( col.m_rows != m_cols || col.m_cols != 1u ) {
		throw std::invalid_argument(
		    "CudaTensor::addColwise(): col must be (cols() x 1), same cols as this tensor" );
	}
	CudaTensor<T> out = *this;
	
	out.addColwiseInPlace( col );

	return out;
}

template <typename T>
CudaTensor<T> &CudaTensor<T>::addColwiseInPlace( CudaTensor const &col )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_data_handle == nullptr || col.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::addColwiseInPlace(): one of the tensors is not initialized" );
	}
	if ( col.m_rows != m_cols || col.m_cols != 1u ) {
		throw std::invalid_argument(
		    "CudaTensor::addColwiseInPlace(): col must be (cols() x 1), same cols as this tensor" );
	}
	cudaError_t cuda_stat = cudaSuccess;
	if constexpr ( std::is_same_v<T, float> ) {
		cuda_stat = cuda_add_colwise_float( m_data_handle, col.m_data_handle, m_rows, m_cols, 1.f );
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cuda_stat = cuda_add_colwise_fp8( m_data_handle, col.m_data_handle, m_rows, m_cols, 1.f );
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::addColwiseInPlace(): unsupported type" );
	}
	if ( cuda_stat != cudaSuccess ) {
		throw std::runtime_error( "CudaTensor::addColwiseInPlace(): device operation failed" );
	}

	return *this;
}

template <typename T>
CudaTensor<T> CudaTensor<T>::subtractColwise( CudaTensor const &col ) const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_data_handle == nullptr || col.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::subtractColwise(): one of the tensors is not initialized" );
	}

	if ( col.m_rows != m_rows || col.m_cols != 1u ) {
		throw std::invalid_argument(
		    "CudaTensor::subtractColwise(): row must be (rows() x 1), same rows as this tensor" );
	}
	cudaError_t cuda_stat = cudaSuccess;
	CudaTensor<T> out = *this;
	if constexpr ( std::is_same_v<T, float> ) {
		cuda_stat = cuda_add_rowwise_float( out.m_data_handle, col.m_data_handle, m_rows, m_cols, -1.f );
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cuda_stat = cuda_add_rowwise_fp8( out.m_data_handle, col.m_data_handle, m_rows, m_cols, -1.f );
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::subtractColwise(): unsupported type" );
	}
	if ( cuda_stat != cudaSuccess ) {
		throw std::runtime_error( "CudaTensor::subtractColwise(): device operation failed" );
	}

	return out;
}

template <typename T>
CudaTensor<T> CudaTensor<T>::operator*( CudaTensor const &other ) const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_rows != other.m_rows || m_cols != other.m_cols ) {
		throw std::invalid_argument( "CudaTensor::operator*(): incompatible dimensions" );
	}
	if ( size() == 0 ) {
		return CudaTensor<T>{};
	}
	if ( m_data_handle == nullptr || other.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::operator*(): one of the tensors is not initialized" );
	}
	CudaTensor<T> out( m_rows, m_cols, value_type{} );
	if constexpr ( std::is_same_v<T, float> ) {
		cudaError_t const err =
		    cuda_mul_float( m_data_handle, other.m_data_handle, out.m_data_handle, size() );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::operator*(): cuda_mul_float failed" );
		}
		return out;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cudaError_t const err =
		    cuda_mul_fp8( m_data_handle, other.m_data_handle, out.m_data_handle, size() );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::operator*(): cuda_mul_fp8 failed" );
		}
		return out;
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::operator*(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> &CudaTensor<T>::operator*=( CudaTensor const &other )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_rows != other.m_rows || m_cols != other.m_cols ) {
		throw std::invalid_argument( "CudaTensor::operator*=(): incompatible dimensions" );
	}
	if ( size() == 0 ) {
		return *this;
	}
	if ( m_data_handle == nullptr || other.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::operator*=(): one of the tensors is not initialized" );
	}
	if constexpr ( std::is_same_v<T, float> ) {
		cudaError_t const err = cuda_mul_float_inplace( m_data_handle, other.m_data_handle, size() );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::operator*=(): cuda_mul_float_inplace failed" );
		}
		return *this;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cudaError_t const err = cuda_mul_fp8_inplace( m_data_handle, other.m_data_handle, size() );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::operator*=(): cuda_mul_fp8_inplace failed" );
		}
		return *this;
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::operator*=(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> CudaTensor<T>::operator*( value_type scalar ) const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( size() == 0 ) {
		return CudaTensor<T>{};
	}
	if ( m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::operator*(): tensor is not initialized" );
	}
	CudaTensor<T> out( *this );
	if constexpr ( std::is_same_v<T, float> ) {
		cudaError_t const err = cuda_scale_float( out.m_data_handle, size(), scalar );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::operator*(): cuda_scale_float failed" );
		}
		return out;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		float const alpha = static_cast<float>( static_cast<__half>( scalar ) );
		cudaError_t const err = cuda_scale_fp8( out.m_data_handle, size(), alpha );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::operator*(): cuda_scale_fp8 failed" );
		}
		return out;
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::operator*(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> &CudaTensor<T>::operator*=( value_type scalar )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( size() == 0 ) {
		return *this;
	}
	if ( m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::operator*=(): tensor is not initialized" );
	}
	if constexpr ( std::is_same_v<T, float> ) {
		cudaError_t const err = cuda_scale_float( m_data_handle, size(), scalar );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::operator*=(): cuda_scale_float failed" );
		}
		return *this;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		float const alpha = static_cast<float>( static_cast<__half>( scalar ) );
		cudaError_t const err = cuda_scale_fp8( m_data_handle, size(), alpha );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::operator*=(): cuda_scale_fp8 failed" );
		}
		return *this;
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::operator*=(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> CudaTensor<T>::cwiseGreater( value_type scalar ) const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( size() == 0 || m_data_handle == nullptr ) {
		return CudaTensor<T>{};
	}
	CudaTensor<T> out( m_rows, m_cols, value_type{} );
	if constexpr ( std::is_same_v<T, float> ) {
		cudaError_t const err =
		    cuda_cwise_greater_float( m_data_handle, out.m_data_handle, size(), static_cast<float>( scalar ) );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::cwiseGreater(): cuda_cwise_greater_float failed" );
		}
		return out;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		float const s = static_cast<float>( static_cast<__half>( scalar ) );
		cudaError_t const err =
		    cuda_cwise_greater_fp8( m_data_handle, out.m_data_handle, size(), s );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::cwiseGreater(): cuda_cwise_greater_fp8 failed" );
		}
		return out;
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::cwiseGreater(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> &CudaTensor<T>::cwiseGreaterInPlace( CudaTensor const &other, value_type scalar )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_data_handle == nullptr || other.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::cwiseGreaterInPlace(): tensor not initialized" );
	}
	if ( m_rows != other.m_rows || m_cols != other.m_cols ) {
		throw std::invalid_argument(
		    "CudaTensor::cwiseGreaterInPlace(other, scalar): incompatible dimensions" );
	}
	if ( size() == 0 ) {
		return *this;
	}
	if constexpr ( std::is_same_v<T, float> ) {
		cudaError_t const err = cuda_cwise_greater_float(
		    other.m_data_handle, m_data_handle, size(), static_cast<float>( scalar ) );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::cwiseGreaterInPlace(): cuda_cwise_greater_float failed" );
		}
		return *this;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		float const s = static_cast<float>( static_cast<__half>( scalar ) );
		cudaError_t const err =
		    cuda_cwise_greater_fp8( other.m_data_handle, m_data_handle, size(), s );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::cwiseGreaterInPlace(): cuda_cwise_greater_fp8 failed" );
		}
		return *this;
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::cwiseGreaterInPlace(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> CudaTensor<T>::cwiseOneMinus() const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( size() == 0 || m_data_handle == nullptr ) {
		return CudaTensor<T>{};
	}
	CudaTensor<T> out( m_rows, m_cols, value_type{} );
	if constexpr ( std::is_same_v<T, float> ) {
		cudaError_t const err = cuda_cwise_one_minus_float( m_data_handle, out.m_data_handle, size() );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::cwiseOneMinus(): cuda_cwise_one_minus_float failed" );
		}
		return out;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cudaError_t const err = cuda_cwise_one_minus_fp8( m_data_handle, out.m_data_handle, size() );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::cwiseOneMinus(): cuda_cwise_one_minus_fp8 failed" );
		}
		return out;
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::cwiseOneMinus(): unsupported type" );
	}
}

template <typename T>
void CudaTensor<T>::cwiseSigmoid( CudaTensor<T> &out ) const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( out.m_rows != m_rows || out.m_cols != m_cols ) {
		out.throw_if_non_owning_realloc(
		    "CudaTensor::cwiseSigmoid(): cannot resize a non-owning output tensor" );
		out = CudaTensor<T>( m_rows, m_cols );
	}
	if ( size() == 0 ) {
		return;
	}
	if ( m_data_handle == nullptr || out.m_data_handle == nullptr ) {
		throw std::runtime_error(
		    "CudaTensor::cwiseSigmoid(): tensor not initialized" );
	}
	if constexpr ( std::is_same_v<T, float> ) {
		cudaError_t const err =
		    cuda_cwise_sigmoid_float( m_data_handle, out.m_data_handle, size() );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::cwiseSigmoid(): cuda_cwise_sigmoid_float failed" );
		}
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cudaError_t const err =
		    cuda_cwise_sigmoid_fp8( m_data_handle, out.m_data_handle, size() );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::cwiseSigmoid(): cuda_cwise_sigmoid_fp8 failed" );
		}
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::cwiseSigmoid(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> CudaTensor<T>::cwiseSigmoid() const
{
	if ( size() == 0 || m_data_handle == nullptr ) {
		return CudaTensor<T>{};
	}
	CudaTensor<T> out;
	cwiseSigmoid( out );
	return out;
}

template <typename T>
CudaTensor<T> CudaTensor<T>::cwiseExp() const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( size() == 0 || m_data_handle == nullptr ) {
		return CudaTensor<T>{};
	}
	CudaTensor<T> out( m_rows, m_cols, value_type{} );
	if constexpr ( std::is_same_v<T, float> ) {
		cudaError_t const err = cuda_cwise_exp_float( m_data_handle, out.m_data_handle, size() );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::cwiseExp(): cuda_cwise_exp_float failed" );
		}
		return out;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cudaError_t const err = cuda_cwise_exp_fp8( m_data_handle, out.m_data_handle, size() );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::cwiseExp(): cuda_cwise_exp_fp8 failed" );
		}
		return out;
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::cwiseExp(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> CudaTensor<T>::cwiseLog() const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( size() == 0 || m_data_handle == nullptr ) {
		return CudaTensor<T>{};
	}
	CudaTensor<T> out( m_rows, m_cols, value_type{} );
	if constexpr ( std::is_same_v<T, float> ) {
		cudaError_t const err = cuda_cwise_log_float( m_data_handle, out.m_data_handle, size() );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::cwiseLog(): cuda_cwise_log_float failed" );
		}
		return out;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cudaError_t const err = cuda_cwise_log_fp8( m_data_handle, out.m_data_handle, size() );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::cwiseLog(): cuda_cwise_log_fp8 failed" );
		}
		return out;
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::cwiseLog(): unsupported type" );
	}
}

template <typename T>
T CudaTensor<T>::sum() const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( size() == 0 || m_data_handle == nullptr ) {
		return value_type{};
	}
	if constexpr ( std::is_same_v<T, float> ) {
		float h = 0.f;
		cudaError_t const err = cuda_sum_float( m_data_handle, size(), &h );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::sum(): cuda_sum_float failed" );
		}
		return h;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		float h = 0.f;
		cudaError_t const err = cuda_sum_fp8( m_data_handle, size(), &h );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::sum(): cuda_sum_fp8 failed" );
		}
		return static_cast<fp8>( static_cast<__half>( h ) );
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::sum(): unsupported type" );
	}
}

template <typename T>
T CudaTensor<T>::asum() const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( size() == 0 || m_data_handle == nullptr ) {
		return value_type{};
	}
	if constexpr ( std::is_same_v<T, float> ) {
		float h = 0.f;
		cublasStatus_t const st = cublasSasum( CublasHandle::instance(), static_cast<int>( size() ),
		                                       m_data_handle, 1, &h );
		if ( st != CUBLAS_STATUS_SUCCESS ) {
			throw std::runtime_error( "CudaTensor::asum(): cublasSasum failed" );
		}
		return h;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		std::vector<fp8> host( size() );
		cudaError_t const cp =
		    cudaMemcpy( host.data(), m_data_handle, size() * sizeof( fp8 ), cudaMemcpyDeviceToHost );
		if ( cp != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::asum(): cudaMemcpy failed" );
		}
		float s = 0.f;
		for ( std::size_t i = 0; i < size(); ++i ) {
			s += fabsf( static_cast<float>( static_cast<__half>( host[i] ) ) );
		}
		return static_cast<fp8>( static_cast<__half>( s ) );
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::asum(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> CudaTensor<T>::sumAlongAxis( std::size_t axis, bool transpose_out ) const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( axis != 0 && axis != 1 ) {
		throw std::invalid_argument( "CudaTensor::sumAlongAxis(): axis must be 0 or 1" );
	}
	if ( size() == 0 || m_data_handle == nullptr ) {
		return CudaTensor<T>{};
	}
	if ( axis == 0 ) {
		CudaTensor<T> out( transpose_out ? m_cols : 1u, transpose_out ? 1u : m_cols, value_type{} );
		if constexpr ( std::is_same_v<T, float> ) {
			cudaError_t const err = cuda_sum_along_axis_float( m_data_handle, out.m_data_handle,
			                                                   m_rows, m_cols, 0, transpose_out );
			if ( err != cudaSuccess ) {
				throw std::runtime_error( "CudaTensor::sumAlongAxis(): cuda_sum_along_axis_float failed" );
			}
			return out;
#if NEURAL_CUDA_ENABLED
		} else if constexpr ( std::is_same_v<T, fp8> ) {
			cudaError_t const err = cuda_sum_along_axis_fp8( m_data_handle, out.m_data_handle,
			                                                 m_rows, m_cols, 0, transpose_out );
			if ( err != cudaSuccess ) {
				throw std::runtime_error( "CudaTensor::sumAlongAxis(): cuda_sum_along_axis_fp8 failed" );
			}
			return out;
#endif
		} else {
			throw std::invalid_argument( "CudaTensor::sumAlongAxis(): unsupported type" );
		}
	}
	CudaTensor<T> out( transpose_out ? 1u : m_rows, transpose_out ? m_rows : 1u, value_type{} );
	if constexpr ( std::is_same_v<T, float> ) {
		cudaError_t const err = cuda_sum_along_axis_float( m_data_handle, out.m_data_handle, m_rows,
		                                                   m_cols, 1, transpose_out );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::sumAlongAxis(): cuda_sum_along_axis_float failed" );
		}
		return out;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cudaError_t const err =
		    cuda_sum_along_axis_fp8( m_data_handle, out.m_data_handle, m_rows, m_cols, 1, transpose_out );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::sumAlongAxis(): cuda_sum_along_axis_fp8 failed" );
		}
		return out;
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::sumAlongAxis(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> &CudaTensor<T>::sumAlongAxisInPlace( CudaTensor const &src, std::size_t axis,
                                                   bool transpose_out )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( this == &src ) {
		throw std::runtime_error(
		    "CudaTensor::sumAlongAxisInPlace(src, axis): src must not be *this" );
	}
	if ( axis != 0 && axis != 1 ) {
		throw std::invalid_argument( "CudaTensor::sumAlongAxisInPlace(): axis must be 0 or 1" );
	}
	if ( m_data_handle == nullptr || src.m_data_handle == nullptr ) {
		throw std::runtime_error( "CudaTensor::sumAlongAxisInPlace(): tensor not initialized" );
	}
	if ( src.size() == 0 ) {
		if ( size() == 0 ) {
			return *this;
		}
		throw std::invalid_argument(
		    "CudaTensor::sumAlongAxisInPlace(): incompatible dimensions" );
	}
	if ( axis == 0 ) {
		bool const ok = transpose_out ? ( m_rows == src.m_cols && m_cols == 1u )
		                              : ( m_rows == 1u && m_cols == src.m_cols );
		if ( !ok ) {
			throw std::invalid_argument(
			    transpose_out
			        ? "CudaTensor::sumAlongAxisInPlace(): for axis 0 with transpose_out, *this must have shape (src.cols(), 1)"
			        : "CudaTensor::sumAlongAxisInPlace(): for axis 0, *this must have shape (1, src.cols())" );
		}
		if constexpr ( std::is_same_v<T, float> ) {
			cudaError_t const err =
			    cuda_sum_along_axis_float( src.m_data_handle, m_data_handle, src.m_rows, src.m_cols, 0,
			                               transpose_out );
			if ( err != cudaSuccess ) {
				throw std::runtime_error(
				    "CudaTensor::sumAlongAxisInPlace(): cuda_sum_along_axis_float failed" );
			}
			return *this;
#if NEURAL_CUDA_ENABLED
		} else if constexpr ( std::is_same_v<T, fp8> ) {
			cudaError_t const err =
			    cuda_sum_along_axis_fp8( src.m_data_handle, m_data_handle, src.m_rows, src.m_cols, 0,
			                             transpose_out );
			if ( err != cudaSuccess ) {
				throw std::runtime_error(
				    "CudaTensor::sumAlongAxisInPlace(): cuda_sum_along_axis_fp8 failed" );
			}
			return *this;
#endif
		} else {
			throw std::invalid_argument( "CudaTensor::sumAlongAxisInPlace(): unsupported type" );
		}
	}
	{
		bool const ok = transpose_out ? ( m_rows == 1u && m_cols == src.m_rows )
		                              : ( m_rows == src.m_rows && m_cols == 1u );
		if ( !ok ) {
			throw std::invalid_argument(
			    transpose_out
			        ? "CudaTensor::sumAlongAxisInPlace(): for axis 1 with transpose_out, *this must have shape (1, src.rows())"
			        : "CudaTensor::sumAlongAxisInPlace(): for axis 1, *this must have shape (src.rows(), 1)" );
		}
	}
	if constexpr ( std::is_same_v<T, float> ) {
		cudaError_t const err =
		    cuda_sum_along_axis_float( src.m_data_handle, m_data_handle, src.m_rows, src.m_cols, 1,
		                               transpose_out );
		if ( err != cudaSuccess ) {
			throw std::runtime_error(
			    "CudaTensor::sumAlongAxisInPlace(): cuda_sum_along_axis_float failed" );
		}
		return *this;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cudaError_t const err =
		    cuda_sum_along_axis_fp8( src.m_data_handle, m_data_handle, src.m_rows, src.m_cols, 1,
		                             transpose_out );
		if ( err != cudaSuccess ) {
			throw std::runtime_error(
			    "CudaTensor::sumAlongAxisInPlace(): cuda_sum_along_axis_fp8 failed" );
		}
		return *this;
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::sumAlongAxisInPlace(): unsupported type" );
	}
}

template <typename T>
CudaTensor<T> CudaTensor<T>::max_along_axis( std::size_t axis ) const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( axis != 0 && axis != 1 ) {
		throw std::invalid_argument( "CudaTensor::max_along_axis(): axis must be 0 or 1" );
	}
	if ( size() == 0 || m_data_handle == nullptr ) {
		return CudaTensor<T>{};
	}
	if ( axis == 0 ) {
		CudaTensor<T> out( 1u, m_cols, value_type{} );
		if constexpr ( std::is_same_v<T, float> ) {
			cudaError_t const err = cuda_max_along_axis_float( m_data_handle, out.m_data_handle,
			                                                   m_rows, m_cols, 0 );
			if ( err != cudaSuccess ) {
				throw std::runtime_error( "CudaTensor::max_along_axis(): cuda_max_along_axis_float failed" );
			}
			return out;
#if NEURAL_CUDA_ENABLED
		} else if constexpr ( std::is_same_v<T, fp8> ) {
			cudaError_t const err = cuda_max_along_axis_fp8( m_data_handle, out.m_data_handle,
			                                                 m_rows, m_cols, 0 );
			if ( err != cudaSuccess ) {
				throw std::runtime_error( "CudaTensor::max_along_axis(): cuda_max_along_axis_fp8 failed" );
			}
			return out;
#endif
		} else {
			throw std::invalid_argument( "CudaTensor::max_along_axis(): unsupported type" );
		}
	}
	CudaTensor<T> out( m_rows, 1u, value_type{} );
	if constexpr ( std::is_same_v<T, float> ) {
		cudaError_t const err =
		    cuda_max_along_axis_float( m_data_handle, out.m_data_handle, m_rows, m_cols, 1 );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::max_along_axis(): cuda_max_along_axis_float failed" );
		}
		return out;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cudaError_t const err =
		    cuda_max_along_axis_fp8( m_data_handle, out.m_data_handle, m_rows, m_cols, 1 );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::max_along_axis(): cuda_max_along_axis_fp8 failed" );
		}
		return out;
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::max_along_axis(): unsupported type" );
	}
}

template <typename T>
CudaTensor<std::uint32_t> CudaTensor<T>::argmaxAlongAxis( std::size_t axis ) const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( axis != 0 && axis != 1 ) {
		throw std::invalid_argument( "CudaTensor::argmaxAlongAxis(): axis must be 0 or 1" );
	}
	if ( size() == 0 || m_data_handle == nullptr ) {
		return CudaTensor<std::uint32_t>{};
	}
	if ( axis == 0 ) {
		CudaTensor<std::uint32_t> out( 1u, m_cols, std::uint32_t{} );
		if constexpr ( std::is_same_v<T, float> ) {
			cudaError_t const err = cuda_argmax_along_axis_float( m_data_handle, out.data(),
			                                                      m_rows, m_cols, 0 );
			if ( err != cudaSuccess ) {
				throw std::runtime_error( "CudaTensor::argmaxAlongAxis(): cuda_argmax_along_axis_float failed" );
			}
			return out;
#if NEURAL_CUDA_ENABLED
		} else if constexpr ( std::is_same_v<T, fp8> ) {
			cudaError_t const err = cuda_argmax_along_axis_fp8( m_data_handle, out.data(),
			                                                    m_rows, m_cols, 0 );
			if ( err != cudaSuccess ) {
				throw std::runtime_error( "CudaTensor::argmaxAlongAxis(): cuda_argmax_along_axis_fp8 failed" );
			}
			return out;
#endif
		} else {
			throw std::invalid_argument( "CudaTensor::argmaxAlongAxis(): unsupported type" );
		}
	}
	CudaTensor<std::uint32_t> out( m_rows, 1u, std::uint32_t{} );
	if constexpr ( std::is_same_v<T, float> ) {
		cudaError_t const err =
		    cuda_argmax_along_axis_float( m_data_handle, out.data(), m_rows, m_cols, 1 );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::argmaxAlongAxis(): cuda_argmax_along_axis_float failed" );
		}
		return out;
#if NEURAL_CUDA_ENABLED
	} else if constexpr ( std::is_same_v<T, fp8> ) {
		cudaError_t const err =
		    cuda_argmax_along_axis_fp8( m_data_handle, out.data(), m_rows, m_cols, 1 );
		if ( err != cudaSuccess ) {
			throw std::runtime_error( "CudaTensor::argmaxAlongAxis(): cuda_argmax_along_axis_fp8 failed" );
		}
		return out;
#endif
	} else {
		throw std::invalid_argument( "CudaTensor::argmaxAlongAxis(): unsupported type" );
	}
}

template <typename T>
template <typename Generator>
void CudaTensor<T>::randomize( Generator &generator ) noexcept
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_data_handle == nullptr || size() == 0 ) {
		return;
	}
	std::vector<T> host( size() );
	std::generate( host.begin(), host.end(), generator );
	cudaMemcpy( m_data_handle, host.data(), size() * sizeof( T ), cudaMemcpyHostToDevice );
}

template <typename T>
void CudaTensor<T>::randomize( std::uint64_t seed ) noexcept
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_data_handle == nullptr || size() == 0 ) {
		return;
	}
	if constexpr ( std::is_same_v<T, float> ) {
		(void)cuda_random_uniform_float( m_data_handle, size(),
		                                 static_cast<unsigned long long>( seed ) );
		return;
	}
	std::mt19937_64 gen( seed );
	std::uniform_real_distribution<double> dis( 0.0, 1.0 );
	auto generator = [&]() { return static_cast<T>( dis( gen ) ); };
	randomize( generator );
}

template <typename T>
void CudaTensor<T>::randomizeHe( std::size_t fan_in, std::uint64_t seed ) noexcept
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( fan_in == 0 ) {
		return;
	}
	if ( m_data_handle == nullptr || size() == 0 ) {
		return;
	}
	if constexpr ( std::is_same_v<T, float> ) {
		double const scale = std::sqrt( 2.0 / static_cast<double>( fan_in ) );
		(void)cuda_random_uniform_symmetric_float(
		    m_data_handle, size(), static_cast<float>( scale ),
		    static_cast<unsigned long long>( seed ) );
		return;
	}
	double const scale = std::sqrt( 2.0 / static_cast<double>( fan_in ) );
	std::mt19937_64 gen( seed );
	std::uniform_real_distribution<double> dis( -scale, scale );
	auto generator = [&]() { return static_cast<T>( dis( gen ) ); };
	randomize( generator );
}

template <typename T>
void CudaTensor<T>::randomizePytorchDefault( std::size_t fan_in, std::uint64_t seed ) noexcept
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( fan_in == 0 ) {
		return;
	}
	if ( m_data_handle == nullptr || size() == 0 ) {
		return;
	}
	if constexpr ( std::is_same_v<T, float> ) {
		float const half_width =
		    static_cast<float>( 1.0 / std::sqrt( static_cast<double>( fan_in ) ) );
		(void)cuda_random_uniform_symmetric_float(
		    m_data_handle, size(), half_width, static_cast<unsigned long long>( seed ) );
		return;
	}
	double const half_width = 1.0 / std::sqrt( static_cast<double>( fan_in ) );
	std::mt19937_64 gen( seed );
	std::uniform_real_distribution<double> dis( -half_width, half_width );
	auto generator = [&]() { return static_cast<T>( dis( gen ) ); };
	randomize( generator );
}

template <typename T>
void CudaTensor<T>::copyToHost( T *host ) const
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	if ( m_data_handle == nullptr || host == nullptr || size() == 0 ) {
		return;
	}
	cudaError_t const err =
	    cudaMemcpy( host, m_data_handle, size() * sizeof( T ), cudaMemcpyDeviceToHost );
	if ( err != cudaSuccess ) {
		throw std::runtime_error( "CudaTensor::copyToHost(): cudaMemcpy failed" );
	}
}

template <typename T, typename Op>
inline void assignBatch(
    CudaTensor<T> &input_batch,
    CudaTensor<T> &target_batch,
    std::vector<Tensor<T>> const &host_inputs,
    std::vector<Tensor<T>> const &host_targets,
    std::vector<int> const &batch_indices_int,
    std::size_t batch_window_start,
    std::size_t batch_size,
    Tensor<T> &host_x,
    Tensor<T> &host_y,
    Op const &op )
{
	CudaStreamSyncOnExit const _cuda_tensor_op_sync;
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for ( std::size_t r = 0; r < batch_size; ++r ) {
		int const idx = batch_indices_int[static_cast<std::size_t>( batch_window_start + r )];
		host_x.assignTensorAsRow( r, host_inputs[static_cast<std::size_t>( idx )] );
		host_y.assignTensorAsRow( r, host_targets[static_cast<std::size_t>( idx )] );
	}
	op( host_x, host_y );
	input_batch.assign( host_x.data(), host_x.size() );
	target_batch.assign( host_y.data(), host_y.size() );
	::neural::cuda_layer_sync();
}


#else // NEURAL_CUDA_ENABLED

template <typename T>
using CudaTensor = Tensor<T>;

#endif // NEURAL_CUDA_ENABLED

} // namespace neural

#endif
