#include "neural_cuda_kernels.hpp"
#include "neural_cuda_runtime.hpp"

namespace neural {

cudaError_t cuda_fill_float( float *, std::size_t, float )
{
	return cudaSuccess;
}

cudaError_t cuda_transpose_float( float const *, float *, std::size_t, std::size_t )
{
	return cudaSuccess;
}

cudaError_t cuda_transpose_fp8( __nv_fp8_e4m3 const *, __nv_fp8_e4m3 *, std::size_t, std::size_t )
{
	return cudaSuccess;
}

cudaError_t cuda_max_abs_float( float const *, std::size_t, float *out_host )
{
	if ( out_host != nullptr ) {
		*out_host = 0.f;
	}
	return cudaSuccess;
}

cudaError_t cuda_max_abs_fp8( __nv_fp8_e4m3 const *, std::size_t, float *out_host )
{
	if ( out_host != nullptr ) {
		*out_host = 0.f;
	}
	return cudaSuccess;
}

cudaError_t cuda_add_fp8( __nv_fp8_e4m3 const *, __nv_fp8_e4m3 const *, __nv_fp8_e4m3 *, std::size_t,
                          float, float )
{
	return cudaSuccess;
}

cudaError_t cuda_add_colwise_float( float *, float const *, std::size_t, std::size_t, float )
{
	return cudaSuccess;
}

cudaError_t cuda_add_rowwise_float( float *, float const *, std::size_t, std::size_t, float )
{
	return cudaSuccess;
}

cudaError_t cuda_add_colwise_fp8( __nv_fp8_e4m3 *, __nv_fp8_e4m3 const *, std::size_t, std::size_t, float )
{
	return cudaSuccess;
}

cudaError_t cuda_add_rowwise_fp8( __nv_fp8_e4m3 *, __nv_fp8_e4m3 const *, std::size_t, std::size_t, float )
{
	return cudaSuccess;
}

cudaError_t cuda_mul_float( float const *, float const *, float *, std::size_t )
{
	return cudaSuccess;
}

cudaError_t cuda_mul_float_inplace( float *, float const *, std::size_t )
{
	return cudaSuccess;
}

cudaError_t cuda_mul_fp8( __nv_fp8_e4m3 const *, __nv_fp8_e4m3 const *, __nv_fp8_e4m3 *, std::size_t )
{
	return cudaSuccess;
}

cudaError_t cuda_mul_fp8_inplace( __nv_fp8_e4m3 *, __nv_fp8_e4m3 const *, std::size_t )
{
	return cudaSuccess;
}

cudaError_t cuda_scale_float( float *, std::size_t, float )
{
	return cudaSuccess;
}

cudaError_t cuda_scale_fp8( __nv_fp8_e4m3 *, std::size_t, float )
{
	return cudaSuccess;
}

cudaError_t cuda_cwise_greater_float( float const *, float *, std::size_t, float )
{
	return cudaSuccess;
}

cudaError_t cuda_cwise_greater_fp8( __nv_fp8_e4m3 const *, __nv_fp8_e4m3 *, std::size_t, float )
{
	return cudaSuccess;
}

cudaError_t cuda_cwise_one_minus_float( float const *, float *, std::size_t )
{
	return cudaSuccess;
}

cudaError_t cuda_cwise_one_minus_fp8( __nv_fp8_e4m3 const *, __nv_fp8_e4m3 *, std::size_t )
{
	return cudaSuccess;
}

cudaError_t cuda_cwise_sigmoid_float( float const *, float *, std::size_t )
{
	return cudaSuccess;
}

cudaError_t cuda_cwise_sigmoid_fp8( __nv_fp8_e4m3 const *, __nv_fp8_e4m3 *, std::size_t )
{
	return cudaSuccess;
}

cudaError_t cuda_cwise_exp_float( float const *, float *, std::size_t )
{
	return cudaSuccess;
}

cudaError_t cuda_cwise_exp_fp8( __nv_fp8_e4m3 const *, __nv_fp8_e4m3 *, std::size_t )
{
	return cudaSuccess;
}

cudaError_t cuda_cwise_log_float( float const *, float *, std::size_t )
{
	return cudaSuccess;
}

cudaError_t cuda_cwise_log_fp8( __nv_fp8_e4m3 const *, __nv_fp8_e4m3 *, std::size_t )
{
	return cudaSuccess;
}

cudaError_t cuda_sum_float( float const *, std::size_t, float *out_host )
{
	if ( out_host != nullptr ) {
		*out_host = 0.f;
	}
	return cudaSuccess;
}

cudaError_t cuda_sum_fp8( __nv_fp8_e4m3 const *, std::size_t, float *out_host )
{
	if ( out_host != nullptr ) {
		*out_host = 0.f;
	}
	return cudaSuccess;
}

cudaError_t cuda_sum_along_axis_float( float const *, float *, std::size_t, std::size_t, int, bool )
{
	return cudaSuccess;
}

cudaError_t cuda_sum_along_axis_fp8( __nv_fp8_e4m3 const *, __nv_fp8_e4m3 *, std::size_t, std::size_t,
                                     int, bool )
{
	return cudaSuccess;
}

cudaError_t cuda_max_along_axis_float( float const *, float *, std::size_t, std::size_t, int )
{
	return cudaSuccess;
}

cudaError_t cuda_max_along_axis_fp8( __nv_fp8_e4m3 const *, __nv_fp8_e4m3 *, std::size_t, std::size_t,
                                     int )
{
	return cudaSuccess;
}

int cuda_device_count()
{
	return -1;
}

bool cuda_runtime_ready()
{
	return false;
}

} // namespace neural
