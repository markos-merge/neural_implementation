#include "cublas_handle.hpp"
#include "neural_cuda_kernels.hpp"
#include "neural_cuda_runtime.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cmath>

namespace neural {

// Short alias for device + host code in this file; same type as __nv_fp8_e4m3.
using fp8 = __nv_fp8_e4m3;

namespace {

__global__ void fill_float_kernel( float *out, unsigned long long n, float value )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		out[i] = value;
	}
}

__global__ void fill_fp8_kernel( fp8 *out, unsigned long long n, fp8 value )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		out[i] = value;
	}
}

__global__ void matmul_fp8_kernel( fp8 *dev, fp8 *other, fp8 *result, unsigned long long rows, unsigned long long other_cols, unsigned long long common_dim )
{
	unsigned long long const row = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const col = static_cast<unsigned long long>( blockIdx.y ) * blockDim.y
	                             + threadIdx.y;
	if ( row < rows && col < other_cols ) {
		__half sum = 0.0;
		for ( unsigned long long k = 0; k < common_dim; k++ ) {
			sum += static_cast<__half>( dev[row * common_dim + k] ) * static_cast<__half>( other[k * other_cols + col] );
		}
		result[row * other_cols + col] = static_cast<fp8>( sum );
	}
}

__global__ void divide_rows_with_col_float_kernel( float *dev, float *other, unsigned long long rows, unsigned long long cols )
{
	unsigned long long const row = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const col = static_cast<unsigned long long>( blockIdx.y ) * blockDim.y
	                             + threadIdx.y;

	if ( row < rows && col < cols ) {
		dev[row * cols + col] /= other[row];
	}
}

__global__ void divide_rows_with_col_fp8_kernel( fp8 *dev, fp8 *other, unsigned long long rows, unsigned long long cols )
{
	unsigned long long const row = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const col = static_cast<unsigned long long>( blockIdx.y ) * blockDim.y
	                             + threadIdx.y;

	if ( row < rows && col < cols ) {
		dev[row * cols + col] = static_cast<fp8>( static_cast<__half>( dev[row * cols + col] ) / static_cast<__half>( other[row] ) );
	}
}

__global__ void max_abs_fp8_reduce_kernel( fp8 const *x, unsigned long long n, float *g_out )
{
	float local = 0.f;
	unsigned long long const stride = static_cast<unsigned long long>( gridDim.x ) * blockDim.x;
	for ( unsigned long long i = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x + threadIdx.x;
	      i < n; i += stride ) {
		float const v = fabsf( static_cast<float>( static_cast<__half>( x[i] ) ) );
		local = fmaxf( local, v );
	}
	__shared__ float smem[256];
	unsigned int const tid = threadIdx.x;
	smem[tid] = local;
	__syncthreads();
	for ( unsigned int s = blockDim.x / 2; s > 0; s >>= 1 ) {
		if ( tid < s ) {
			smem[tid] = fmaxf( smem[tid], smem[tid + s] );
		}
		__syncthreads();
	}
	if ( tid == 0 ) {
		atomicMax( reinterpret_cast<unsigned int *>( g_out ), __float_as_uint( smem[0] ) );
	}
}

__global__ void add_fp8_kernel( fp8 const *a, fp8 const *b, fp8 *out, unsigned long long n, float alpha,
                                float beta )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		float const va = static_cast<float>( static_cast<__half>( a[i] ) );
		float const vb = static_cast<float>( static_cast<__half>( b[i] ) );
		float const vc = alpha * va + beta * vb;
		out[i] = static_cast<fp8>( static_cast<__half>( vc ) );
	}
}

__global__ void add_colwise_float_kernel( float *dev, float const *col_vec, unsigned long long rows,
                                          unsigned long long cols, float alpha )
{
	unsigned long long const row = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const col = static_cast<unsigned long long>( blockIdx.y ) * blockDim.y
	                             + threadIdx.y;
	
	if ( row < rows && col < cols ) {
		dev[row * cols + col] += alpha * col_vec[col];
	}
}

__global__ void add_rowwise_float_kernel( float *dev, float const *row_vec, unsigned long long rows,
                                          unsigned long long cols, float alpha )
{
	unsigned long long const row = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const col = static_cast<unsigned long long>( blockIdx.y ) * blockDim.y
	                             + threadIdx.y;
	if ( row < rows && col < cols ) {
		dev[row * cols + col] += alpha * row_vec[row];
	}
}

__global__ void add_colwise_fp8_kernel( fp8 *dev, fp8 const *col_vec, unsigned long long rows,
                                        unsigned long long cols, float alpha )
{
	unsigned long long const row = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const col = static_cast<unsigned long long>( blockIdx.y ) * blockDim.y
	                             + threadIdx.y;
	
	if ( row < rows && col < cols ) {
		float const v = static_cast<float>( static_cast<__half>( dev[row * cols + col] ) )
		              + alpha * static_cast<float>( static_cast<__half>( col_vec[col] ) );
		dev[row * cols + col] = static_cast<fp8>( static_cast<__half>( v ) );
	}
}

__global__ void add_rowwise_fp8_kernel( fp8 *dev, fp8 const *row_vec, unsigned long long rows,
                                        unsigned long long cols, float alpha )
{
	unsigned long long const row = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const col = static_cast<unsigned long long>( blockIdx.y ) * blockDim.y
	                             + threadIdx.y;
	if ( row < rows && col < cols ) {
		float const v = static_cast<float>( static_cast<__half>( dev[row * cols + col] ) )
		              + alpha * static_cast<float>( static_cast<__half>( row_vec[row] ) );
		dev[row * cols + col] = static_cast<fp8>( static_cast<__half>( v ) );
	}
}

__global__ void mul_float_kernel( float const *a, float const *b, float *out, unsigned long long n )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		out[i] = a[i] * b[i];
	}
}

__global__ void mul_float_inplace_kernel( float *a, float const *b, unsigned long long n )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		a[i] *= b[i];
	}
}

__global__ void mul_fp8_kernel( fp8 const *a, fp8 const *b, fp8 *out, unsigned long long n )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		float const p = static_cast<float>( static_cast<__half>( a[i] ) )
		                * static_cast<float>( static_cast<__half>( b[i] ) );
		out[i] = static_cast<fp8>( static_cast<__half>( p ) );
	}
}

__global__ void mul_fp8_inplace_kernel( fp8 *a, fp8 const *b, unsigned long long n )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		float const p = static_cast<float>( static_cast<__half>( a[i] ) )
		                * static_cast<float>( static_cast<__half>( b[i] ) );
		a[i] = static_cast<fp8>( static_cast<__half>( p ) );
	}
}

__global__ void scale_fp8_kernel( fp8 *dev, unsigned long long n, float alpha )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		float const p = static_cast<float>( static_cast<__half>( dev[i] ) ) * alpha;
		dev[i] = static_cast<fp8>( static_cast<__half>( p ) );
	}
}

__global__ void sum_float_atomic_kernel( float const *in, unsigned long long n, float *g_out )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	float local = 0.f;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		local += in[i];
	}
	atomicAdd( g_out, local );
}

__global__ void sum_fp8_atomic_kernel( fp8 const *in, unsigned long long n, float *g_out )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	float local = 0.f;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		local += static_cast<float>( static_cast<__half>( in[i] ) );
	}
	atomicAdd( g_out, local );
}

__global__ void cwise_greater_float_kernel( float const *in, float *out, unsigned long long n,
                                            float s )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		out[i] = ( in[i] > s ) ? 1.f : 0.f;
	}
}

__global__ void cwise_greater_fp8_kernel( fp8 const *in, fp8 *out, unsigned long long n, float s )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		float const v = static_cast<float>( static_cast<__half>( in[i] ) );
		out[i] = static_cast<fp8>( static_cast<__half>( ( v > s ) ? 1.f : 0.f ) );
	}
}

__global__ void cwise_one_minus_float_kernel( float const *in, float *out, unsigned long long n )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		out[i] = 1.f - in[i];
	}
}

__global__ void cwise_one_minus_fp8_kernel( fp8 const *in, fp8 *out, unsigned long long n )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		float const v = static_cast<float>( static_cast<__half>( in[i] ) );
		out[i] = static_cast<fp8>( static_cast<__half>( 1.f - v ) );
	}
}

__global__ void cwise_sigmoid_float_kernel( float const *in, float *out, unsigned long long n )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		float const x = in[i];
		out[i] = 1.f / ( 1.f + expf( -x ) );
	}
}

__global__ void cwise_sigmoid_fp8_kernel( fp8 const *in, fp8 *out, unsigned long long n )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		float const x = static_cast<float>( static_cast<__half>( in[i] ) );
		float const y = 1.f / ( 1.f + expf( -x ) );
		out[i] = static_cast<fp8>( static_cast<__half>( y ) );
	}
}

__global__ void cwise_exp_float_kernel( float const *in, float *out, unsigned long long n )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		out[i] = expf( in[i] );
	}
}

__global__ void cwise_exp_fp8_kernel( fp8 const *in, fp8 *out, unsigned long long n )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		float const v = expf( static_cast<float>( static_cast<__half>( in[i] ) ) );
		out[i] = static_cast<fp8>( static_cast<__half>( v ) );
	}
}

__global__ void cwise_log_float_kernel( float const *in, float *out, unsigned long long n )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		out[i] = logf( in[i] );
	}
}

__global__ void cwise_log_fp8_kernel( fp8 const *in, fp8 *out, unsigned long long n )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		float const v = logf( static_cast<float>( static_cast<__half>( in[i] ) ) );
		out[i] = static_cast<fp8>( static_cast<__half>( v ) );
	}
}

__global__ void sum_axis0_float_kernel( float const *in, float *out, unsigned long long rows,
                                        unsigned long long cols )
{
	unsigned long long const c = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                           + threadIdx.x;
	if ( c < cols ) {
		float s = 0.f;
		for ( unsigned long long r = 0; r < rows; ++r ) {
			s += in[r * cols + c];
		}
		out[c] = s;
	}
}

__global__ void sum_axis1_float_kernel( float const *in, float *out, unsigned long long rows,
                                        unsigned long long cols )
{
	unsigned long long const r = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                           + threadIdx.x;
	if ( r < rows ) {
		float s = 0.f;
		for ( unsigned long long c = 0; c < cols; ++c ) {
			s += in[r * cols + c];
		}
		out[r] = s;
	}
}

__global__ void max_axis0_float_kernel( float const *in, float *out, unsigned long long rows,
                                        unsigned long long cols )
{
	unsigned long long const c = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                           + threadIdx.x;
	if ( c < cols ) {
		float m = -INFINITY;
		for ( unsigned long long r = 0; r < rows; ++r ) {
			m = fmaxf( m, in[r * cols + c] );
		}
		out[c] = m;
	}
}

__global__ void max_axis1_float_kernel( float const *in, float *out, unsigned long long rows,
                                        unsigned long long cols )
{
	unsigned long long const r = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                           + threadIdx.x;
	if ( r < rows ) {
		float m = -INFINITY;
		for ( unsigned long long c = 0; c < cols; ++c ) {
			m = fmaxf( m, in[r * cols + c] );
		}
		out[r] = m;
	}
}

__global__ void sum_axis0_fp8_kernel( fp8 const *in, fp8 *out, unsigned long long rows,
                                       unsigned long long cols )
{
	unsigned long long const c = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                           + threadIdx.x;
	if ( c < cols ) {
		float s = 0.f;
		for ( unsigned long long r = 0; r < rows; ++r ) {
			s += static_cast<float>( static_cast<__half>( in[r * cols + c] ) );
		}
		out[c] = static_cast<fp8>( static_cast<__half>( s ) );
	}
}

__global__ void sum_axis1_fp8_kernel( fp8 const *in, fp8 *out, unsigned long long rows,
                                     unsigned long long cols )
{
	unsigned long long const r = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                           + threadIdx.x;
	if ( r < rows ) {
		float s = 0.f;
		for ( unsigned long long c = 0; c < cols; ++c ) {
			s += static_cast<float>( static_cast<__half>( in[r * cols + c] ) );
		}
		out[r] = static_cast<fp8>( static_cast<__half>( s ) );
	}
}

__global__ void max_axis0_fp8_kernel( fp8 const *in, fp8 *out, unsigned long long rows,
                                     unsigned long long cols )
{
	unsigned long long const c = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                           + threadIdx.x;
	if ( c < cols ) {
		float m = -INFINITY;
		for ( unsigned long long r = 0; r < rows; ++r ) {
			m = fmaxf( m, static_cast<float>( static_cast<__half>( in[r * cols + c] ) ) );
		}
		out[c] = static_cast<fp8>( static_cast<__half>( m ) );
	}
}

__global__ void max_axis1_fp8_kernel( fp8 const *in, fp8 *out, unsigned long long rows,
                                     unsigned long long cols )
{
	unsigned long long const r = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                           + threadIdx.x;
	if ( r < rows ) {
		float m = -INFINITY;
		for ( unsigned long long c = 0; c < cols; ++c ) {
			m = fmaxf( m, static_cast<float>( static_cast<__half>( in[r * cols + c] ) ) );
		}
		out[r] = static_cast<fp8>( static_cast<__half>( m ) );
	}
}
} // namespace

cudaError_t cuda_fill_float( float *dev, std::size_t count, float value )
{
	if ( count == 0 || dev == nullptr ) {
		return cudaSuccess;
	}

	int const threads = 256;
	int const blocks = static_cast<int>( ( count + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	fill_float_kernel<<<blocks, threads>>>( dev, static_cast<unsigned long long>( count ), value );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}

	return cudaDeviceSynchronize();
}

cudaError_t cuda_fill_fp8( fp8 *dev, std::size_t count, fp8 value )
{
	if ( count == 0 || dev == nullptr ) {
		return cudaSuccess;
	}

	int const threads = 256;
	int const blocks = static_cast<int>( ( count + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	fill_fp8_kernel<<<blocks, threads>>>( dev, static_cast<unsigned long long>( count ), value );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}

	return cudaDeviceSynchronize();
}

__global__ void transpose_float_kernel( float const *in, float *out, unsigned long long rows,
                                        unsigned long long cols )
{
	unsigned long long const r = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                           + threadIdx.x;
	unsigned long long const c = static_cast<unsigned long long>( blockIdx.y ) * blockDim.y
	                           + threadIdx.y;
	if ( r < rows && c < cols ) {
		out[c * rows + r] = in[r * cols + c];
	}
}

__global__ void transpose_fp8_kernel( fp8 const *in, fp8 *out, unsigned long long rows, unsigned long long cols )
{
	unsigned long long const r = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                           + threadIdx.x;
	unsigned long long const c = static_cast<unsigned long long>( blockIdx.y ) * blockDim.y
	                           + threadIdx.y;
	if ( r < rows && c < cols ) {
		out[c * rows + r] = in[r * cols + c];
	}
}

cudaError_t cuda_transpose_float( float const *in, float *out, std::size_t rows, std::size_t cols )
{
	if ( rows == 0 || cols == 0 ) {
		return cudaSuccess;
	}
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	dim3 threads( 16, 16, 1 );
	dim3 blocks( static_cast<int>( ( static_cast<std::size_t>( rows ) + threads.x - 1 ) / threads.x ),
	             static_cast<int>( ( static_cast<std::size_t>( cols ) + threads.y - 1 ) / threads.y ),
	             1 );
	transpose_float_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( rows ),
	                                              static_cast<unsigned long long>( cols ) );
	return cudaGetLastError();
}

cudaError_t cuda_transpose_fp8( fp8 const *in, fp8 *out, std::size_t rows, std::size_t cols )
{
	if ( rows == 0 || cols == 0 ) {
		return cudaSuccess;
	}
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	dim3 threads( 16, 16, 1 );
	dim3 blocks( static_cast<int>( ( static_cast<std::size_t>( rows ) + threads.x - 1 ) / threads.x ),
	             static_cast<int>( ( static_cast<std::size_t>( cols ) + threads.y - 1 ) / threads.y ),
	             1 );
	transpose_fp8_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( rows ),
	                                           static_cast<unsigned long long>( cols ) );
	return cudaGetLastError();
}

cudaError_t cuda_matmul_fp8( fp8 *dev, fp8 *other, fp8 *result, std::size_t rows, std::size_t cols, std::size_t other_rows, std::size_t other_cols )
{
	if ( rows == 0 || cols == 0 || other_rows == 0 || other_cols == 0 ) {
		return cudaSuccess;
	}
	
	dim3 threads(16, 16, 1);
	dim3 blocks( static_cast< int >( (static_cast< std::size_t >( rows ) + threads.x - 1) / threads.x ), static_cast< int >( (static_cast< std::size_t >( other_cols ) + threads.y - 1) / threads.y ), 1 );
	matmul_fp8_kernel<<<blocks, threads>>>( dev, other, result, static_cast<unsigned long long>( rows ), static_cast<unsigned long long>( other_cols ), static_cast<unsigned long long>( other_rows ) );
	return cudaGetLastError();
}

cudaError_t cuda_matmul_inplace_fp8( fp8 *dev, fp8 *other, std::size_t rows, std::size_t cols, std::size_t other_rows, std::size_t other_cols )
{
	if ( rows == 0 || cols == 0 || other_rows == 0 || other_cols == 0 ) {
		return cudaSuccess;
	}
	
	
	dim3 threads(16, 16, 1);
	dim3 blocks( static_cast< int >( (static_cast< std::size_t >( rows ) + threads.x - 1) / threads.x ), static_cast< int >( (static_cast< std::size_t >( other_cols ) + threads.y - 1) / threads.y ), 1 );
	matmul_fp8_kernel<<<blocks, threads>>>( dev, other, dev, static_cast<unsigned long long>( rows ), static_cast<unsigned long long>( other_cols ), static_cast<unsigned long long>( other_rows ) );
	return cudaGetLastError();
}

cudaError_t divide_rows_with_col_float( float *dev, float *other, std::size_t rows, std::size_t cols )
{
	if ( rows == 0 || cols == 0 ) {
		return cudaSuccess;
	}
	
	dim3 threads(16, 16, 1);
	dim3 blocks( static_cast< int >( (static_cast< std::size_t >( rows ) + threads.x - 1) / threads.x ), static_cast< int >( (static_cast< std::size_t >( cols ) + threads.y - 1) / threads.y ), 1 );
	divide_rows_with_col_float_kernel<<<blocks, threads>>>( dev, other, static_cast<unsigned long long>( rows ), static_cast<unsigned long long>( cols ) );
	return cudaGetLastError();
}

cudaError_t divide_rows_with_col_fp8( fp8 *dev, fp8 *other, std::size_t rows, std::size_t cols )
{
	if ( rows == 0 || cols == 0 ) {
		return cudaSuccess;
	}

	dim3 threads( 16, 16, 1 );
	dim3 blocks( static_cast<int>( ( static_cast<std::size_t>( rows ) + threads.x - 1 ) / threads.x ),
	             static_cast<int>( ( static_cast<std::size_t>( cols ) + threads.y - 1 ) / threads.y ),
	             1 );
	divide_rows_with_col_fp8_kernel<<<blocks, threads>>>(
	    dev, other, static_cast<unsigned long long>( rows ), static_cast<unsigned long long>( cols ) );
	return cudaGetLastError();
}

cudaError_t cuda_max_abs_float( float const *dev, std::size_t count, float *out_host )
{
	if ( out_host == nullptr ) {
		return cudaErrorInvalidValue;
	}
	if ( count == 0 || dev == nullptr ) {
		*out_host = 0.f;
		return cudaSuccess;
	}
	int idx = 0;
	cublasStatus_t const st = cublasIsamax(
	    CublasHandle::instance(), static_cast<int>( count ), dev, 1, &idx );
	if ( st != CUBLAS_STATUS_SUCCESS ) {
		return cudaErrorUnknown;
	}
	cudaError_t err = cudaDeviceSynchronize();
	if ( err != cudaSuccess ) {
		return err;
	}
	float v = 0.f;
	err = cudaMemcpy( &v, dev + static_cast<std::size_t>( idx - 1 ), sizeof( float ),
	                  cudaMemcpyDeviceToHost );
	if ( err != cudaSuccess ) {
		return err;
	}
	*out_host = std::fabs( v );
	return cudaSuccess;
}

cudaError_t cuda_max_abs_fp8( fp8 const *dev, std::size_t count, float *out_host )
{
	if ( out_host == nullptr ) {
		return cudaErrorInvalidValue;
	}
	if ( count == 0 || dev == nullptr ) {
		*out_host = 0.f;
		return cudaSuccess;
	}
	float *d_max = nullptr;
	cudaError_t err = cudaMalloc( reinterpret_cast<void **>( &d_max ), sizeof( float ) );
	if ( err != cudaSuccess ) {
		return err;
	}
	err = cudaMemset( d_max, 0, sizeof( float ) );
	if ( err != cudaSuccess ) {
		cudaFree( d_max );
		return err;
	}
	int const threads = 256;
	int blocks = static_cast<int>(
	    ( count + static_cast<std::size_t>( threads ) - 1 ) / static_cast<std::size_t>( threads ) );
	if ( blocks < 1 ) {
		blocks = 1;
	}
	if ( blocks > 65535 ) {
		blocks = 65535;
	}
	max_abs_fp8_reduce_kernel<<<blocks, threads>>>( dev, static_cast<unsigned long long>( count ),
	                                                  d_max );
	err = cudaGetLastError();
	if ( err != cudaSuccess ) {
		cudaFree( d_max );
		return err;
	}
	err = cudaMemcpy( out_host, d_max, sizeof( float ), cudaMemcpyDeviceToHost );
	cudaFree( d_max );
	if ( err != cudaSuccess ) {
		return err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_add_fp8( fp8 const *a, fp8 const *b, fp8 *out, std::size_t count, float alpha,
                           float beta )
{
	if ( count == 0 ) {
		return cudaSuccess;
	}
	if ( a == nullptr || b == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( count + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	add_fp8_kernel<<<blocks, threads>>>(
	    a, b, out, static_cast<unsigned long long>( count ), alpha, beta );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_add_colwise_float( float *dev, float const *col_vec, std::size_t rows, std::size_t cols,
                                    float alpha )
{
	if ( rows == 0 || cols == 0 ) {
		return cudaSuccess;
	}
	
	if ( dev == nullptr || col_vec == nullptr ) {
		return cudaErrorInvalidValue;
	}
	
	dim3 threads(16, 16, 1);
	dim3 blocks( static_cast< int >( (static_cast< std::size_t >( rows ) + threads.x - 1) / threads.x ), static_cast< int >( (static_cast< std::size_t >( cols ) + threads.y - 1) / threads.y ), 1 );
	add_colwise_float_kernel<<<blocks, threads>>>( dev, col_vec, static_cast<unsigned long long>( rows ), static_cast<unsigned long long>( cols ), alpha );

	return cudaGetLastError();
}

cudaError_t cuda_add_colwise_fp8( fp8 *dev, fp8 const *col_vec, std::size_t rows, std::size_t cols,
                                  float alpha )
{
	if ( rows == 0 || cols == 0 ) {
		return cudaSuccess;
	}
	
	if ( dev == nullptr || col_vec == nullptr ) {
		return cudaErrorInvalidValue;
	}
	
	dim3 threads(16, 16, 1);
	dim3 blocks( static_cast< int >( (static_cast< std::size_t >( rows ) + threads.x - 1) / threads.x ), static_cast< int >( (static_cast< std::size_t >( cols ) + threads.y - 1) / threads.y ), 1 );
	add_colwise_fp8_kernel<<<blocks, threads>>>( dev, col_vec, static_cast<unsigned long long>( rows ), static_cast<unsigned long long>( cols ), alpha );
	return cudaGetLastError();
}

cudaError_t cuda_add_rowwise_float( float *dev, float const *row_vec, std::size_t rows, std::size_t cols,
                                    float alpha )
{
	if ( rows == 0 || cols == 0 ) {
		return cudaSuccess;
	}
	if ( dev == nullptr || row_vec == nullptr ) {
		return cudaErrorInvalidValue;
	}
	dim3 threads( 16, 16, 1 );
	dim3 blocks( static_cast<int>( ( static_cast<std::size_t>( rows ) + threads.x - 1 ) / threads.x ),
	             static_cast<int>( ( static_cast<std::size_t>( cols ) + threads.y - 1 ) / threads.y ),
	             1 );
	add_rowwise_float_kernel<<<blocks, threads>>>(
	    dev, row_vec, static_cast<unsigned long long>( rows ), static_cast<unsigned long long>( cols ),
	    alpha );
	return cudaGetLastError();
}

cudaError_t cuda_add_rowwise_fp8( fp8 *dev, fp8 const *row_vec, std::size_t rows, std::size_t cols,
                                  float alpha )
{
	if ( rows == 0 || cols == 0 ) {
		return cudaSuccess;
	}
	if ( dev == nullptr || row_vec == nullptr ) {
		return cudaErrorInvalidValue;
	}
	dim3 threads( 16, 16, 1 );
	dim3 blocks( static_cast<int>( ( static_cast<std::size_t>( rows ) + threads.x - 1 ) / threads.x ),
	             static_cast<int>( ( static_cast<std::size_t>( cols ) + threads.y - 1 ) / threads.y ),
	             1 );
	add_rowwise_fp8_kernel<<<blocks, threads>>>(
	    dev, row_vec, static_cast<unsigned long long>( rows ), static_cast<unsigned long long>( cols ),
	    alpha );
	return cudaGetLastError();
}

cudaError_t cuda_mul_float( float const *a, float const *b, float *out, std::size_t count )
{
	if ( count == 0 ) {
		return cudaSuccess;
	}
	if ( a == nullptr || b == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( count + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	mul_float_kernel<<<blocks, threads>>>( a, b, out, static_cast<unsigned long long>( count ) );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_mul_float_inplace( float *a, float const *b, std::size_t count )
{
	if ( count == 0 ) {
		return cudaSuccess;
	}
	if ( a == nullptr || b == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( count + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	mul_float_inplace_kernel<<<blocks, threads>>>( a, b, static_cast<unsigned long long>( count ) );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_mul_fp8( fp8 const *a, fp8 const *b, fp8 *out, std::size_t count )
{
	if ( count == 0 ) {
		return cudaSuccess;
	}
	if ( a == nullptr || b == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( count + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	mul_fp8_kernel<<<blocks, threads>>>( a, b, out, static_cast<unsigned long long>( count ) );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_mul_fp8_inplace( fp8 *a, fp8 const *b, std::size_t count )
{
	if ( count == 0 ) {
		return cudaSuccess;
	}
	if ( a == nullptr || b == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( count + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	mul_fp8_inplace_kernel<<<blocks, threads>>>( a, b, static_cast<unsigned long long>( count ) );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_scale_float( float *dev, std::size_t count, float alpha )
{
	if ( count == 0 ) {
		return cudaSuccess;
	}
	if ( dev == nullptr ) {
		return cudaErrorInvalidValue;
	}
	cublasStatus_t const st = cublasSscal(
	    CublasHandle::instance(), static_cast<int>( count ), &alpha, dev, 1 );
	if ( st != CUBLAS_STATUS_SUCCESS ) {
		return cudaErrorUnknown;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_scale_fp8( fp8 *dev, std::size_t count, float alpha )
{
	if ( count == 0 ) {
		return cudaSuccess;
	}
	if ( dev == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( count + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	scale_fp8_kernel<<<blocks, threads>>>( dev, static_cast<unsigned long long>( count ), alpha );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_sum_float( float const *in, std::size_t n, float *out_host )
{
	if ( out_host == nullptr ) {
		return cudaErrorInvalidValue;
	}
	if ( n == 0 || in == nullptr ) {
		*out_host = 0.f;
		return cudaSuccess;
	}
	float *d_sum = nullptr;
	cudaError_t err = cudaMalloc( reinterpret_cast<void **>( &d_sum ), sizeof( float ) );
	if ( err != cudaSuccess ) {
		return err;
	}
	err = cudaMemset( d_sum, 0, sizeof( float ) );
	if ( err != cudaSuccess ) {
		cudaFree( d_sum );
		return err;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( n + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	int const capped = blocks > 65535 ? 65535 : blocks;
	sum_float_atomic_kernel<<<capped, threads>>>( in, static_cast<unsigned long long>( n ), d_sum );
	err = cudaGetLastError();
	if ( err != cudaSuccess ) {
		cudaFree( d_sum );
		return err;
	}
	err = cudaDeviceSynchronize();
	if ( err != cudaSuccess ) {
		cudaFree( d_sum );
		return err;
	}
	err = cudaMemcpy( out_host, d_sum, sizeof( float ), cudaMemcpyDeviceToHost );
	cudaFree( d_sum );
	return err;
}

cudaError_t cuda_sum_fp8( fp8 const *in, std::size_t n, float *out_host )
{
	if ( out_host == nullptr ) {
		return cudaErrorInvalidValue;
	}
	if ( n == 0 || in == nullptr ) {
		*out_host = 0.f;
		return cudaSuccess;
	}
	float *d_sum = nullptr;
	cudaError_t err = cudaMalloc( reinterpret_cast<void **>( &d_sum ), sizeof( float ) );
	if ( err != cudaSuccess ) {
		return err;
	}
	err = cudaMemset( d_sum, 0, sizeof( float ) );
	if ( err != cudaSuccess ) {
		cudaFree( d_sum );
		return err;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( n + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	int const capped = blocks > 65535 ? 65535 : blocks;
	sum_fp8_atomic_kernel<<<capped, threads>>>( in, static_cast<unsigned long long>( n ), d_sum );
	err = cudaGetLastError();
	if ( err != cudaSuccess ) {
		cudaFree( d_sum );
		return err;
	}
	err = cudaDeviceSynchronize();
	if ( err != cudaSuccess ) {
		cudaFree( d_sum );
		return err;
	}
	err = cudaMemcpy( out_host, d_sum, sizeof( float ), cudaMemcpyDeviceToHost );
	cudaFree( d_sum );
	return err;
}

cudaError_t cuda_cwise_greater_float( float const *in, float *out, std::size_t n, float scalar )
{
	if ( n == 0 ) {
		return cudaSuccess;
	}
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( n + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	cwise_greater_float_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( n ),
	                                                  scalar );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_cwise_greater_fp8( fp8 const *in, fp8 *out, std::size_t n, float scalar )
{
	if ( n == 0 ) {
		return cudaSuccess;
	}
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( n + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	cwise_greater_fp8_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( n ),
	                                                scalar );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_cwise_one_minus_float( float const *in, float *out, std::size_t n )
{
	if ( n == 0 ) {
		return cudaSuccess;
	}
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( n + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	cwise_one_minus_float_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( n ) );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_cwise_one_minus_fp8( fp8 const *in, fp8 *out, std::size_t n )
{
	if ( n == 0 ) {
		return cudaSuccess;
	}
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( n + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	cwise_one_minus_fp8_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( n ) );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_cwise_sigmoid_float( float const *in, float *out, std::size_t n )
{
	if ( n == 0 ) {
		return cudaSuccess;
	}
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( n + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	cwise_sigmoid_float_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( n ) );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_cwise_sigmoid_fp8( fp8 const *in, fp8 *out, std::size_t n )
{
	if ( n == 0 ) {
		return cudaSuccess;
	}
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( n + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	cwise_sigmoid_fp8_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( n ) );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_cwise_exp_float( float const *in, float *out, std::size_t n )
{
	if ( n == 0 ) {
		return cudaSuccess;
	}
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( n + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	cwise_exp_float_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( n ) );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_cwise_exp_fp8( fp8 const *in, fp8 *out, std::size_t n )
{
	if ( n == 0 ) {
		return cudaSuccess;
	}
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( n + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	cwise_exp_fp8_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( n ) );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_cwise_log_float( float const *in, float *out, std::size_t n )
{
	if ( n == 0 ) {
		return cudaSuccess;
	}
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( n + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	cwise_log_float_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( n ) );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_cwise_log_fp8( fp8 const *in, fp8 *out, std::size_t n )
{
	if ( n == 0 ) {
		return cudaSuccess;
	}
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( n + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	cwise_log_fp8_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( n ) );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_sum_along_axis_float( float const *in, float *out, std::size_t rows, std::size_t cols,
                                       int axis )
{
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	if ( axis == 0 ) {
		if ( cols == 0 ) {
			return cudaSuccess;
		}
		int const blocks = static_cast<int>( ( cols + static_cast<std::size_t>( threads ) - 1 )
		                                     / static_cast<std::size_t>( threads ) );
		sum_axis0_float_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( rows ),
		                                              static_cast<unsigned long long>( cols ) );
	} else if ( axis == 1 ) {
		if ( rows == 0 ) {
			return cudaSuccess;
		}
		int const blocks = static_cast<int>( ( rows + static_cast<std::size_t>( threads ) - 1 )
		                                     / static_cast<std::size_t>( threads ) );
		sum_axis1_float_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( rows ),
		                                              static_cast<unsigned long long>( cols ) );
	} else {
		return cudaErrorInvalidValue;
	}
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_sum_along_axis_fp8( fp8 const *in, fp8 *out, std::size_t rows, std::size_t cols,
                                     int axis )
{
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	if ( axis == 0 ) {
		if ( cols == 0 ) {
			return cudaSuccess;
		}
		int const blocks = static_cast<int>( ( cols + static_cast<std::size_t>( threads ) - 1 )
		                                     / static_cast<std::size_t>( threads ) );
		sum_axis0_fp8_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( rows ),
		                                            static_cast<unsigned long long>( cols ) );
	} else if ( axis == 1 ) {
		if ( rows == 0 ) {
			return cudaSuccess;
		}
		int const blocks = static_cast<int>( ( rows + static_cast<std::size_t>( threads ) - 1 )
		                                     / static_cast<std::size_t>( threads ) );
		sum_axis1_fp8_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( rows ),
		                                            static_cast<unsigned long long>( cols ) );
	} else {
		return cudaErrorInvalidValue;
	}
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_max_along_axis_float( float const *in, float *out, std::size_t rows, std::size_t cols,
                                       int axis )
{
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	if ( axis == 0 ) {
		if ( cols == 0 ) {
			return cudaSuccess;
		}
		int const blocks = static_cast<int>( ( cols + static_cast<std::size_t>( threads ) - 1 )
		                                     / static_cast<std::size_t>( threads ) );
		max_axis0_float_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( rows ),
		                                              static_cast<unsigned long long>( cols ) );
	} else if ( axis == 1 ) {
		if ( rows == 0 ) {
			return cudaSuccess;
		}
		int const blocks = static_cast<int>( ( rows + static_cast<std::size_t>( threads ) - 1 )
		                                     / static_cast<std::size_t>( threads ) );
		max_axis1_float_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( rows ),
		                                              static_cast<unsigned long long>( cols ) );
	} else {
		return cudaErrorInvalidValue;
	}
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

cudaError_t cuda_max_along_axis_fp8( fp8 const *in, fp8 *out, std::size_t rows, std::size_t cols,
                                     int axis )
{
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	if ( axis == 0 ) {
		if ( cols == 0 ) {
			return cudaSuccess;
		}
		int const blocks = static_cast<int>( ( cols + static_cast<std::size_t>( threads ) - 1 )
		                                     / static_cast<std::size_t>( threads ) );
		max_axis0_fp8_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( rows ),
		                                            static_cast<unsigned long long>( cols ) );
	} else if ( axis == 1 ) {
		if ( rows == 0 ) {
			return cudaSuccess;
		}
		int const blocks = static_cast<int>( ( rows + static_cast<std::size_t>( threads ) - 1 )
		                                     / static_cast<std::size_t>( threads ) );
		max_axis1_fp8_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( rows ),
		                                            static_cast<unsigned long long>( cols ) );
	} else {
		return cudaErrorInvalidValue;
	}
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaDeviceSynchronize();
}

int cuda_device_count()
{
	int n = 0;
	cudaError_t const err = cudaGetDeviceCount( &n );
	if ( err != cudaSuccess ) {
		return -1;
	}
	return n;
}

bool cuda_runtime_ready()
{
	int n = 0;
	if ( cudaGetDeviceCount( &n ) != cudaSuccess || n <= 0 ) {
		return false;
	}
	void *p = nullptr;
	cudaError_t const err = cudaMalloc( &p, sizeof( float ) );
	if ( err != cudaSuccess ) {
		return false;
	}
	cudaFree( p );
	return true;
}

} // namespace neural
