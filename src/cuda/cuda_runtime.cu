#include "cublas_handle.hpp"
#include "neural_cuda_kernels.hpp"
#include "neural_cuda_runtime.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <curand.h>
#include <algorithm>
#include <cmath>
#include <device_types.h>
#include <mutex>
#include <atomic>
#include <cstdio>

namespace {
std::atomic<bool> g_cuda_post_kernel_synchronize{ false };
struct CudaOpSyncGuard
{
	~CudaOpSyncGuard()
	{
		if ( !g_cuda_post_kernel_synchronize.load( std::memory_order_relaxed ) ) {
			return;
		}
		(void)cudaGetLastError();
		cudaError_t const e = cudaDeviceSynchronize();
		if ( e != cudaSuccess ) {
			std::fprintf( stderr, "neural: cudaDeviceSynchronize (post-kernel): %s\n", cudaGetErrorString( e ) );
		}
	}
};
} // namespace

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
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		a[i] *= b[i];
	}
}

__global__ void bn_running_var_from_saved_inv_std_kernel( float *running_var, float const *saved_inv_std,
                                                          unsigned long long channels, float eps )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < channels; i += stride ) {
		float const inv = saved_inv_std[i];
		running_var[i] = 1.f / ( inv * inv ) - eps;
	}
}

__global__ void bn_saved_inv_std_from_variance_kernel( float *saved_inv, float const *var,
                                                     unsigned long long channels, float eps )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < channels; i += stride ) {
		saved_inv[i] = rsqrtf( var[i] + eps );
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

__global__ void argmax_axis0_float_kernel( float const *in, unsigned int *out, unsigned long long rows,
                                           unsigned long long cols )
{
	unsigned long long const c = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                           + threadIdx.x;
	if ( c < cols ) {
		float m = in[c];
		unsigned int best = 0;
		for ( unsigned long long r = 1; r < rows; ++r ) {
			float const v = in[r * cols + c];
			if ( v > m ) {
				m = v;
				best = static_cast<unsigned int>( r );
			}
		}
		out[c] = best;
	}
}

__global__ void argmax_axis1_float_kernel( float const *in, unsigned int *out, unsigned long long rows,
                                          unsigned long long cols )
{
	unsigned long long const r = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                           + threadIdx.x;
	if ( r < rows ) {
		float m = in[r * cols];
		unsigned int best = 0;
		for ( unsigned long long c = 1; c < cols; ++c ) {
			float const v = in[r * cols + c];
			if ( v > m ) {
				m = v;
				best = static_cast<unsigned int>( c );
			}
		}
		out[r] = best;
	}
}

__global__ void argmax_axis0_fp8_kernel( fp8 const *in, unsigned int *out, unsigned long long rows,
                                        unsigned long long cols )
{
	unsigned long long const c = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( c < cols ) {
		float m = static_cast<float>( static_cast<__half>( in[c] ) );
		unsigned int best = 0;
		for ( unsigned long long r = 1; r < rows; ++r ) {
			float const v = static_cast<float>( static_cast<__half>( in[r * cols + c] ) );
			if ( v > m ) {
				m = v;
				best = static_cast<unsigned int>( r );
			}
		}
		out[c] = best;
	}
}

__global__ void argmax_axis1_fp8_kernel( fp8 const *in, unsigned int *out, unsigned long long rows,
                                        unsigned long long cols )
{
	unsigned long long const r = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                           + threadIdx.x;
	if ( r < rows ) {
		float m = static_cast<float>( static_cast<__half>( in[r * cols] ) );
		unsigned int best = 0;
		for ( unsigned long long c = 1; c < cols; ++c ) {
			float const v = static_cast<float>( static_cast<__half>( in[r * cols + c] ) );
			if ( v > m ) {
				m = v;
				best = static_cast<unsigned int>( c );
			}
		}
		out[r] = best;
	}
}
} // namespace

cudaError_t cuda_fill_float( float *dev, std::size_t count, float value )
{
	CudaOpSyncGuard const _cuda_op_sync;
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

	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_fill_fp8( fp8 *dev, std::size_t count, fp8 value )
{
	CudaOpSyncGuard const _cuda_op_sync;
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

	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

__global__ void uniform_symmetric_inplace_kernel( float *x, unsigned long long n, float half_width )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	float const scale = 2.f * half_width;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		x[i] = x[i] * scale - half_width;
	}
}

namespace detail {

std::mutex g_curand_mutex;
curandGenerator_t g_curand_generator = nullptr;

cudaError_t curand_generate_uniform_chunks( curandGenerator_t gen, float *dev, std::size_t count )
{
	CudaOpSyncGuard const _cuda_op_sync;
	constexpr std::size_t kMaxChunk = static_cast<std::size_t>( 1 ) << 28;
	for ( std::size_t off = 0; off < count; off += kMaxChunk ) {
		std::size_t const chunk = std::min( kMaxChunk, count - off );
		curandStatus_t const st = curandGenerateUniform( gen, dev + off, chunk );
		if ( st != CURAND_STATUS_SUCCESS ) {
			return cudaErrorUnknown;
		}
	}
	return cudaSuccess;
}

/// Pre: \c g_curand_mutex held, generator created.
/// Reseeds and resets the absolute offset so the same \p seed always starts the same stream;
/// without the offset reset the singleton generator's offset persists across calls and the
/// "same seed → same output" contract breaks for callers like \c DropoutLayer.
cudaError_t cuda_random_uniform_float_unlocked( float *dev, std::size_t count, unsigned long long seed )
{
	CudaOpSyncGuard const _cuda_op_sync;
	curandStatus_t st = curandSetPseudoRandomGeneratorSeed( g_curand_generator, seed );
	if ( st != CURAND_STATUS_SUCCESS ) {
		return cudaErrorUnknown;
	}
	st = curandSetGeneratorOffset( g_curand_generator, 0ULL );
	if ( st != CURAND_STATUS_SUCCESS ) {
		return cudaErrorUnknown;
	}
	return curand_generate_uniform_chunks( g_curand_generator, dev, count );
}

} // namespace detail

cudaError_t cuda_random_uniform_float( float *dev, std::size_t count, unsigned long long seed )
{
	CudaOpSyncGuard const _cuda_op_sync;
	if ( count == 0 ) {
		return cudaSuccess;
	}
	if ( dev == nullptr ) {
		return cudaErrorInvalidValue;
	}
	std::lock_guard<std::mutex> lock( detail::g_curand_mutex );
	if ( detail::g_curand_generator == nullptr ) {
		curandStatus_t const cr =
		    curandCreateGenerator( &detail::g_curand_generator, CURAND_RNG_PSEUDO_PHILOX4_32_10 );
		if ( cr != CURAND_STATUS_SUCCESS ) {
			return cudaErrorInitializationError;
		}
	}
	return detail::cuda_random_uniform_float_unlocked( dev, count, seed );
}

cudaError_t cuda_random_uniform_symmetric_float( float *dev, std::size_t count, float half_width,
                                                 unsigned long long seed )
{
	CudaOpSyncGuard const _cuda_op_sync;
	if ( count == 0 ) {
		return cudaSuccess;
	}
	if ( dev == nullptr ) {
		return cudaErrorInvalidValue;
	}
	std::lock_guard<std::mutex> lock( detail::g_curand_mutex );
	if ( detail::g_curand_generator == nullptr ) {
		curandStatus_t const cr =
		    curandCreateGenerator( &detail::g_curand_generator, CURAND_RNG_PSEUDO_PHILOX4_32_10 );
		if ( cr != CURAND_STATUS_SUCCESS ) {
			return cudaErrorInitializationError;
		}
	}
	cudaError_t err = detail::cuda_random_uniform_float_unlocked( dev, count, seed );
	if ( err != cudaSuccess ) {
		return err;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( count + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	uniform_symmetric_inplace_kernel<<<blocks, threads>>>(
	    dev, static_cast<unsigned long long>( count ), half_width );
	err = cudaGetLastError();
	return err;
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

__global__ void gather_rows_float_kernel( float const * __restrict__ src, float * __restrict__ dst,
                                        int const * __restrict__ row_indices, unsigned long long cols,
                                        unsigned long long dst_rows )
{
	unsigned long long const total = dst_rows * cols;
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long flat = idx; flat < total; flat += stride ) {
		unsigned long long const r = flat / cols;
		unsigned long long const c = flat % cols;
		int const src_row = row_indices[r];
		dst[flat] = src[static_cast<unsigned long long>( src_row ) * cols + c];
	}
}

__global__ void gather_rows_fp8_kernel( fp8 const * __restrict__ src, fp8 * __restrict__ dst,
                                       int const * __restrict__ row_indices, unsigned long long cols,
                                       unsigned long long dst_rows )
{
	unsigned long long const total = dst_rows * cols;
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x
	                             + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long flat = idx; flat < total; flat += stride ) {
		unsigned long long const r = flat / cols;
		unsigned long long const c = flat % cols;
		int const src_row = row_indices[r];
		dst[flat] = src[static_cast<unsigned long long>( src_row ) * cols + c];
	}
}

cudaError_t cuda_gather_rows_float( float const *src, float *dst, std::size_t src_rows, std::size_t cols,
                                    int const *indices_host, std::size_t row_indices_src,
                                    std::size_t row_indices_size )
{
	CudaOpSyncGuard const _cuda_op_sync;
	(void)src_rows;
	if ( row_indices_size == 0u || cols == 0u ) {
		return cudaSuccess;
	}
	if ( src == nullptr || dst == nullptr || indices_host == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int *d_indices = nullptr;
	cudaError_t err = cudaMalloc( &d_indices, row_indices_size * sizeof( int ) );
	if ( err != cudaSuccess ) {
		return err;
	}
	err = cudaMemcpy( d_indices, indices_host + row_indices_src, row_indices_size * sizeof( int ),
	                  cudaMemcpyHostToDevice );
	if ( err != cudaSuccess ) {
		cudaFree( d_indices );
		return err;
	}
	unsigned long long const total = static_cast<unsigned long long>( row_indices_size )
	                               * static_cast<unsigned long long>( cols );
	unsigned int const block_size = 256u;
	unsigned long long const blocks64 = ( total + static_cast<unsigned long long>( block_size ) - 1ULL )
	                                  / static_cast<unsigned long long>( block_size );
	unsigned int grid = static_cast<unsigned int>( std::min( blocks64, 65535ull ) );
	if ( grid < 1u ) {
		grid = 1u;
	}
	gather_rows_float_kernel<<<grid, block_size>>>( src, dst, d_indices,
	    static_cast<unsigned long long>( cols ),
	    static_cast<unsigned long long>( row_indices_size ) );
	err = cudaGetLastError();
	if ( err != cudaSuccess ) {
		cudaFree( d_indices );
		return err;
	}
	// err = cudaDeviceSynchronize();
	err = cudaSuccess;
	cudaFree( d_indices );
	return err;
}

cudaError_t cuda_gather_rows_fp8( fp8 const *src, fp8 *dst, std::size_t src_rows, std::size_t cols,
                                  int const *indices_host, std::size_t row_indices_src,
                                  std::size_t row_indices_size )
{
	CudaOpSyncGuard const _cuda_op_sync;
	(void)src_rows;
	if ( row_indices_size == 0u || cols == 0u ) {
		return cudaSuccess;
	}
	if ( src == nullptr || dst == nullptr || indices_host == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int *d_indices = nullptr;
	cudaError_t err = cudaMalloc( &d_indices, row_indices_size * sizeof( int ) );
	if ( err != cudaSuccess ) {
		return err;
	}
	err = cudaMemcpy( d_indices, indices_host + row_indices_src, row_indices_size * sizeof( int ),
	                  cudaMemcpyHostToDevice );
	if ( err != cudaSuccess ) {
		cudaFree( d_indices );
		return err;
	}
	unsigned long long const total = static_cast<unsigned long long>( row_indices_size )
	                               * static_cast<unsigned long long>( cols );
	unsigned int const block_size = 256u;
	unsigned long long const blocks64 = ( total + static_cast<unsigned long long>( block_size ) - 1ULL )
	                                  / static_cast<unsigned long long>( block_size );
	unsigned int grid = static_cast<unsigned int>( std::min( blocks64, 65535ull ) );
	if ( grid < 1u ) {
		grid = 1u;
	}
	gather_rows_fp8_kernel<<<grid, block_size>>>( src, dst, d_indices,
	    static_cast<unsigned long long>( cols ),
	    static_cast<unsigned long long>( row_indices_size ) );
	err = cudaGetLastError();
	if ( err != cudaSuccess ) {
		cudaFree( d_indices );
		return err;
	}
	// err = cudaDeviceSynchronize();
	cudaFree( d_indices );
	return err;
}

cudaError_t cuda_transpose_float( float const *in, float *out, std::size_t rows, std::size_t cols )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
	CudaOpSyncGuard const _cuda_op_sync;
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
	CudaOpSyncGuard const _cuda_op_sync;
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
	CudaOpSyncGuard const _cuda_op_sync;
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
	CudaOpSyncGuard const _cuda_op_sync;
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
	CudaOpSyncGuard const _cuda_op_sync;
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
	CudaOpSyncGuard const _cuda_op_sync;
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
	// cudaError_t err = cudaDeviceSynchronize();
	cudaError_t err = cudaSuccess;
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
	CudaOpSyncGuard const _cuda_op_sync;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_add_fp8( fp8 const *a, fp8 const *b, fp8 *out, std::size_t count, float alpha,
                           float beta )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_add_colwise_float( float *dev, float const *col_vec, std::size_t rows, std::size_t cols,
                                    float alpha )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
	CudaOpSyncGuard const _cuda_op_sync;
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
	CudaOpSyncGuard const _cuda_op_sync;
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
	CudaOpSyncGuard const _cuda_op_sync;
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

cudaError_t cuda_bn_running_var_from_saved_inv_std_float( float *running_var, float const *saved_inv_std,
                                                          std::size_t channels, float eps )
{
	CudaOpSyncGuard const _cuda_op_sync;
	if ( channels == 0u ) {
		return cudaSuccess;
	}
	if ( running_var == nullptr || saved_inv_std == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( channels + static_cast<std::size_t>( threads ) - 1u )
	                                   / static_cast<std::size_t>( threads ) );
	bn_running_var_from_saved_inv_std_kernel<<<blocks, threads>>>(
	    running_var, saved_inv_std, static_cast<unsigned long long>( channels ), eps );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaSuccess;
}

cudaError_t cuda_bn_saved_inv_std_from_variance_float( float *saved_inv, float const *var,
                                                       std::size_t channels, float eps )
{
	CudaOpSyncGuard const _cuda_op_sync;
	if ( channels == 0u ) {
		return cudaSuccess;
	}
	if ( saved_inv == nullptr || var == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( channels + static_cast<std::size_t>( threads ) - 1u )
	                                   / static_cast<std::size_t>( threads ) );
	bn_saved_inv_std_from_variance_kernel<<<blocks, threads>>>(
	    saved_inv, var, static_cast<unsigned long long>( channels ), eps );
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaSuccess;
}

cudaError_t cuda_mul_float( float const *a, float const *b, float *out, std::size_t count )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_mul_float_inplace( float *a, float const *b, std::size_t count )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_mul_fp8( fp8 const *a, fp8 const *b, fp8 *out, std::size_t count )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
	CudaOpSyncGuard const _cuda_op_sync;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_scale_float( float *dev, std::size_t count, float alpha )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_scale_fp8( fp8 *dev, std::size_t count, float alpha )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_sum_float( float const *in, std::size_t n, float *out_host )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
	// err = cudaDeviceSynchronize();
	err = cudaSuccess;
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
	CudaOpSyncGuard const _cuda_op_sync;
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
	// err = cudaDeviceSynchronize();
	err = cudaSuccess;
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
	CudaOpSyncGuard const _cuda_op_sync;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_cwise_greater_fp8( fp8 const *in, fp8 *out, std::size_t n, float scalar )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_cwise_one_minus_float( float const *in, float *out, std::size_t n )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_cwise_one_minus_fp8( fp8 const *in, fp8 *out, std::size_t n )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_cwise_sigmoid_float( float const *in, float *out, std::size_t n )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_cwise_sigmoid_fp8( fp8 const *in, fp8 *out, std::size_t n )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_cwise_exp_float( float const *in, float *out, std::size_t n )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
	CudaOpSyncGuard const _cuda_op_sync;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_cwise_log_float( float const *in, float *out, std::size_t n )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_cwise_log_fp8( fp8 const *in, fp8 *out, std::size_t n )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_sum_along_axis_float( float const *in, float *out, std::size_t rows, std::size_t cols,
                                       int axis, bool transpose_out )
{
	CudaOpSyncGuard const _cuda_op_sync;
	(void)transpose_out;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_sum_along_axis_fp8( fp8 const *in, fp8 *out, std::size_t rows, std::size_t cols,
                                     int axis, bool transpose_out )
{
	CudaOpSyncGuard const _cuda_op_sync;
	(void)transpose_out;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_max_along_axis_float( float const *in, float *out, std::size_t rows, std::size_t cols,
                                       int axis )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_max_along_axis_fp8( fp8 const *in, fp8 *out, std::size_t rows, std::size_t cols,
                                     int axis )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
	// return cudaDeviceSynchronize();
	return cudaSuccess;
}

cudaError_t cuda_argmax_along_axis_float( float const *in, unsigned int *out, std::size_t rows,
                                          std::size_t cols, int axis )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
		argmax_axis0_float_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( rows ),
		                                                static_cast<unsigned long long>( cols ) );
	} else if ( axis == 1 ) {
		if ( rows == 0 ) {
			return cudaSuccess;
		}
		int const blocks = static_cast<int>( ( rows + static_cast<std::size_t>( threads ) - 1 )
		                                     / static_cast<std::size_t>( threads ) );
		argmax_axis1_float_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( rows ),
		                                                static_cast<unsigned long long>( cols ) );
	} else {
		return cudaErrorInvalidValue;
	}
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaSuccess;
}

cudaError_t cuda_argmax_along_axis_fp8( fp8 const *in, unsigned int *out, std::size_t rows,
                                        std::size_t cols, int axis )
{
	CudaOpSyncGuard const _cuda_op_sync;
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
		argmax_axis0_fp8_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( rows ),
		                                              static_cast<unsigned long long>( cols ) );
	} else if ( axis == 1 ) {
		if ( rows == 0 ) {
			return cudaSuccess;
		}
		int const blocks = static_cast<int>( ( rows + static_cast<std::size_t>( threads ) - 1 )
		                                     / static_cast<std::size_t>( threads ) );
		argmax_axis1_fp8_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( rows ),
		                                              static_cast<unsigned long long>( cols ) );
	} else {
		return cudaErrorInvalidValue;
	}
	cudaError_t const launch_err = cudaGetLastError();
	if ( launch_err != cudaSuccess ) {
		return launch_err;
	}
	return cudaSuccess;
}

__global__ void mul_substract_float_kernel( float *data, float const *other,
                                            unsigned long long count, float scalar )
{
	unsigned long long const idx    = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < count; i += stride ) {
		data[i] -= other[i] * scalar;
	}
}

cudaError_t cuda_mul_substract_float( float *data, float const *other, std::size_t count,
                                      float scalar )
{
	CudaOpSyncGuard const _cuda_op_sync;
	if ( data == nullptr || other == nullptr || count == 0 ) {
		return cudaSuccess;
	}
	int const threads = 256;
	int const blocks  = static_cast<int>(
	    ( count + static_cast<std::size_t>( threads ) - 1 ) / static_cast<std::size_t>( threads ) );
	mul_substract_float_kernel<<<blocks, threads>>>( data, other,
	                                                  static_cast<unsigned long long>( count ),
	                                                  scalar );
	cudaError_t const err = cudaGetLastError();
	return err != cudaSuccess ? err : cudaSuccess;
}

/// col2im: accumulate im2col columns back into image tensor (used in conv backward).
/// col layout : [N, C_in, Kh, Kw, out_H, out_W]  (standard im2col column-major patch order)
/// im  layout : [N, C_in, H, W]
/// Valid convolution: out_H = H - Kh + 1,  out_W = W - Kw + 1
__global__ void col2im_float_kernel( float const *col, float *im,
                                     int N, int C_in, int H, int W, int Kh, int Kw )
{
	int const out_H   = H - Kh + 1;
	int const out_W   = W - Kw + 1;
	int const total   = N * C_in * H * W;
	int const idx     = blockIdx.x * blockDim.x + threadIdx.x;
	int const stride  = blockDim.x * gridDim.x;

	for ( int i = idx; i < total; i += stride ) {
		int const w_im  = i % W;
		int const h_im  = ( i / W ) % H;
		int const c     = ( i / ( W * H ) ) % C_in;
		int const n     = i / ( W * H * C_in );

		float val = 0.f;
		for ( int kh = 0; kh < Kh; ++kh ) {
			int const oh = h_im - kh;
			if ( oh < 0 || oh >= out_H ) {
				continue;
			}
			for ( int kw = 0; kw < Kw; ++kw ) {
				int const ow = w_im - kw;
				if ( ow < 0 || ow >= out_W ) {
					continue;
				}
				// col index: [n, c, kh, kw, oh, ow]
				int const col_idx = ( ( ( ( n * C_in + c ) * Kh + kh ) * Kw + kw ) * out_H + oh ) * out_W + ow;
				val += col[col_idx];
			}
		}
		im[i] += val;
	}
}

__global__ void add_elementwise_float_kernel( float *in, float *out, unsigned long long n, float value )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	for ( unsigned long long i = idx; i < n; i += stride ) {
		out[i] = in[i] + value;
	}
}

__global__ void add_elementwise_fp8_kernel( fp8 *in, fp8 *out, unsigned long long n, fp8 value )
{
	unsigned long long const idx = static_cast<unsigned long long>( blockIdx.x ) * blockDim.x + threadIdx.x;
	unsigned long long const stride = static_cast<unsigned long long>( blockDim.x ) * gridDim.x;
	float const vadd = static_cast<float>( static_cast<__half>( value ) );
	for ( unsigned long long i = idx; i < n; i += stride ) {
		float const v = static_cast<float>( static_cast<__half>( in[i] ) ) + vadd;
		out[i] = static_cast<fp8>( static_cast<__half>( v ) );
	}
}

cudaError_t cuda_add_elementwise( float *in, float *out, std::size_t n, float value )
{
	CudaOpSyncGuard const _cuda_op_sync;
	if ( n == 0 ) {
		return cudaSuccess;
	}
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( n + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );

	add_elementwise_float_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( n ),
	                                                   value );
	return cudaGetLastError();
}

cudaError_t cuda_add_elementwise( fp8 *in, fp8 *out, std::size_t n, fp8 value )
{
	CudaOpSyncGuard const _cuda_op_sync;
	if ( n == 0 ) {
		return cudaSuccess;
	}
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	int const threads = 256;
	int const blocks = static_cast<int>( ( n + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );

	add_elementwise_fp8_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( n ),
	                                                 value );
	return cudaGetLastError();
}

const int threads_per_block = 32*32;
__global__ void reduce_sum_to_dim_float_kernel( float *in, float *out,
                                                unsigned long long prefix,
                                                unsigned long long mid,
                                                unsigned long long suffix )
{
	__shared__ float partial_sums[threads_per_block];

	unsigned long long const idx = blockIdx.x;
	if ( idx < mid ) {
		unsigned long long tidx = threadIdx.x;
		unsigned long long nx = blockDim.x;
		unsigned long long tidy = threadIdx.y;
		unsigned long long ny = blockDim.y;
		unsigned long long start_i = ( prefix*tidx )/nx;
		unsigned long long end_i = min( ( prefix*(tidx+1) )/nx, prefix );
		unsigned long long start_j = ( suffix*tidy )/ny;
		unsigned long long end_j = min( ( suffix*(tidy+1) )/ny, suffix );
		float partial_sum = 0.f;
		for ( unsigned long long i = start_i; i < end_i; ++i ) {
			for ( unsigned long long j = start_j; j < end_j; ++j ) {
				partial_sum += in[i * mid * suffix + idx * suffix + j];
			}
		}
		partial_sums[tidx*blockDim.y + tidy] = partial_sum;
	} else {
		partial_sums[threadIdx.x * blockDim.y + threadIdx.y] = 0.f;
	}
	__syncthreads();
	if ( threadIdx.x == 0 && threadIdx.y == 0 ) {
		float total = 0.f;
		for ( int i = 0; i < threads_per_block; ++i ) {
			total += partial_sums[i];
		}
		if ( idx < mid ) {
			out[idx] = total;
		}
	}
}

cudaError_t cuda_reduce_sum_to_dim( float *in, float *out, std::size_t prefix, std::size_t mid,
                                    std::size_t suffix )
{
	CudaOpSyncGuard const _cuda_op_sync;
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	if ( prefix == 0 || mid == 0 || suffix == 0 ) {
		return cudaSuccess;
	}

	dim3 const threads( 32, 32, 1 );
	dim3 const blocks( static_cast<unsigned>( mid ), 1U, 1U );

	reduce_sum_to_dim_float_kernel<<<blocks, threads>>>(
	    in, out, static_cast<unsigned long long>( prefix ), static_cast<unsigned long long>( mid ),
	    static_cast<unsigned long long>( suffix ) );

	return cudaGetLastError();
}

__global__ void reduce_sum_to_dim_axis_0_float_kernel( float *in, float *out,
                                                        unsigned long long n,
                                                        unsigned long long stride )
{
	__shared__ float partial_sums[threads_per_block];
	unsigned long long const idx = blockIdx.x;
	unsigned long long const tidx = threadIdx.x;
	unsigned long long const nx = blockDim.x;
	unsigned long long const start_i = ( stride*tidx )/nx;
	unsigned long long const end_i = min( ( stride*(tidx+1) )/nx, stride );
	float partial_sum = 0.f;
	for ( unsigned long long i = start_i; i < end_i; ++i ) {
		partial_sum += in[idx * stride + i];
	}
	partial_sums[tidx] = partial_sum;
	__syncthreads();
	if ( tidx == 0 ) {
		float total = 0.f;
		for ( int i = 0; i < threads_per_block; ++i ) {
			total += partial_sums[i];
		}
		out[idx] = total;
	}
}

cudaError_t cuda_reduce_sum_to_dim_axis_0( float *in, float *out, std::size_t n, std::size_t stride )
{
	CudaOpSyncGuard const _cuda_op_sync;
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	if ( n == 0 || stride == 0 ) {
		return cudaSuccess;
	}

	int const threads = threads_per_block;
	int const blocks = static_cast<int>( ( n + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	reduce_sum_to_dim_axis_0_float_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( n ),
	                                                             static_cast<unsigned long long>( stride ) );
	return cudaGetLastError();
}

__global__ void swap_axes_float_kernel( float *in, float *out, unsigned long long n,
                                        unsigned long *stride,
                                        unsigned long *new_stride,
                                        unsigned long long rank,
                                        unsigned long const *new_axes )
{
	constexpr unsigned long long mx_rank = 10;
	unsigned long long const idx = blockIdx.x;
	unsigned long long const tidx = threadIdx.x;
	unsigned long long index[mx_rank] = { 0 };
	unsigned long long global_tid = idx*blockDim.x*gridDim.x + tidx;
	unsigned long long global_threads = blockDim.x*gridDim.x;
	
	for( unsigned long i = global_tid; i < n; i += global_threads ) {
		unsigned long long i_copy = i;
		for( unsigned long j = 0; j + 1 < rank; ++j ) {
			index[j] = i_copy / stride[j];
			i_copy %= stride[j];
		}
		index[rank-1] = i_copy;

		unsigned long long new_index = 0;
		for( unsigned long j = 0; j < rank; ++j ) {
			new_index += index[new_axes[j]] * new_stride[j];
		}
		out[new_index] = in[i];
	}
}

cudaError_t cuda_swap_axes( float *in, float *out, std::size_t n, std::size_t *strides, std::size_t *new_strides, std::size_t rank, const std::size_t *new_axes )
{
	CudaOpSyncGuard const _cuda_op_sync;
	//todo: this can be done better if a thread could use more than one matrix
	if ( in == nullptr || out == nullptr ) {
		return cudaErrorInvalidValue;
	}
	if ( n == 0 || strides == nullptr || new_strides == nullptr ) {
		return cudaSuccess;
	}

	int const threads = threads_per_block;
	int const blocks = static_cast< int >( ( n + static_cast<std::size_t>( threads ) - 1 )
	                                   / static_cast<std::size_t>( threads ) );
	static_assert( sizeof( std::size_t ) == sizeof( unsigned long ),
	               "cuda_swap_axes expects std::size_t to match unsigned long long" );
	swap_axes_float_kernel<<<blocks, threads>>>( in, out, static_cast<unsigned long long>( n ),
	                                             static_cast<unsigned long *>( strides ),
	                                             static_cast<unsigned long *>( new_strides ),
	                                             static_cast<unsigned long long>( rank ), 
	                                             reinterpret_cast<unsigned long const *>( new_axes ) );
	return cudaGetLastError();
}

cudaError_t cuda_col2im_float( float const *col, float *im, int N, int C_in, int H, int W,
                               int Kh, int Kw )
{
	CudaOpSyncGuard const _cuda_op_sync;
	if ( col == nullptr || im == nullptr ) {
		return cudaSuccess;
	}
	int const total   = N * C_in * H * W;
	int const threads = 256;
	int const blocks  = ( total + threads - 1 ) / threads;
	col2im_float_kernel<<<blocks, threads>>>( col, im, N, C_in, H, W, Kh, Kw );
	cudaError_t const err = cudaGetLastError();
	return err != cudaSuccess ? err : cudaSuccess;
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
