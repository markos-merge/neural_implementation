#ifndef NEURAL_IMPL_NEURAL_CUDA_KERNELS_HPP
#define NEURAL_IMPL_NEURAL_CUDA_KERNELS_HPP

#include <cstddef>

#if NEURAL_CUDA_ENABLED
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#else
using cudaError_t = int;
enum : cudaError_t
{
	cudaSuccess = 0,
};
/// Placeholder when CUDA is disabled so FP8-related declarations still type-check.
struct __nv_fp8_e4m3
{
};
#endif

namespace neural {

/// Fills `count` floats on the device starting at `dev` with `value`.
/// Returns `cudaSuccess` on success (or when `count == 0`).
cudaError_t cuda_fill_float( float *dev, std::size_t count, float value );
/// Fills `count` FP8 (E4M3) elements on the device.
cudaError_t cuda_fill_fp8( __nv_fp8_e4m3 *dev, std::size_t count, __nv_fp8_e4m3 value );

/// Row-major transpose: `out` is `cols×rows`, `out[j,i] = in[i,j]`.
cudaError_t cuda_transpose_float( float const *in, float *out, std::size_t rows, std::size_t cols );
cudaError_t cuda_transpose_fp8( __nv_fp8_e4m3 const *in, __nv_fp8_e4m3 *out, std::size_t rows, std::size_t cols );
cudaError_t cuda_swap_axes( float *in, float *out, std::size_t n, std::size_t *strides,
                            std::size_t *new_strides, std::size_t rank,
                            std::size_t const *new_axes );

/// Row-major gather in one kernel: `dst` has `row_indices_size` rows; for each output row `r`,
/// `dst(r,:) = src(indices_host[row_indices_src + r],:)`. `src_rows` is unused but kept for symmetry
/// with callers that validate against it.
cudaError_t cuda_gather_rows_float( float const *src, float *dst, std::size_t src_rows, std::size_t cols,
                                   int const *indices_host, std::size_t row_indices_src,
                                   std::size_t row_indices_size );
cudaError_t cuda_gather_rows_fp8( __nv_fp8_e4m3 const *src, __nv_fp8_e4m3 *dst, std::size_t src_rows,
                                  std::size_t cols, int const *indices_host, std::size_t row_indices_src,
                                  std::size_t row_indices_size );

cudaError_t cuda_matmul_fp8( __nv_fp8_e4m3 *dev, __nv_fp8_e4m3 *other, __nv_fp8_e4m3 *result, std::size_t rows, std::size_t cols, std::size_t other_rows, std::size_t other_cols );
/// Same as `cuda_matmul_fp8` but result is written back into `dev` (in-place).
cudaError_t cuda_matmul_inplace_fp8( __nv_fp8_e4m3 *dev, __nv_fp8_e4m3 *other, std::size_t rows, std::size_t cols,
                                     std::size_t other_rows, std::size_t other_cols );

cudaError_t divide_rows_with_col_float( float *dev, float *other, std::size_t rows, std::size_t cols );
cudaError_t divide_rows_with_col_fp8( __nv_fp8_e4m3 *dev, __nv_fp8_e4m3 *other, std::size_t rows, std::size_t cols );

/// Writes the largest absolute value to `*out_host` (0 if `count == 0`).
cudaError_t cuda_max_abs_float( float const *dev, std::size_t count, float *out_host );
cudaError_t cuda_max_abs_fp8( __nv_fp8_e4m3 const *dev, std::size_t count, float *out_host );

/// Element-wise: `out[i] = alpha * a[i] + beta * b[i]` (float path uses `cublasSgeam` in CudaTensor).
cudaError_t cuda_add_fp8( __nv_fp8_e4m3 const *a, __nv_fp8_e4m3 const *b, __nv_fp8_e4m3 *out,
                          std::size_t count, float alpha, float beta );

/// Row-major `dev` is `rows x cols`; `col` is `cols x 1` (bias per output column). In-place:
/// `dev(r,c) += alpha * col(c,0)`.
cudaError_t cuda_add_colwise_float( float *dev, float const *col, std::size_t rows, std::size_t cols,
                                    float alpha );
cudaError_t cuda_add_colwise_fp8( __nv_fp8_e4m3 *dev, __nv_fp8_e4m3 const *col, std::size_t rows,
                                  std::size_t cols, float alpha );

/// Row-major `dev` is `rows x cols`; `row` is `rows x 1` (one value per batch row). In-place:
/// `dev(r,c) += alpha * row(r,0)` (softmax shift).
cudaError_t cuda_add_rowwise_float( float *dev, float const *row_vec, std::size_t rows, std::size_t cols,
                                    float alpha );
cudaError_t cuda_add_rowwise_fp8( __nv_fp8_e4m3 *dev, __nv_fp8_e4m3 const *row_vec, std::size_t rows,
                                  std::size_t cols, float alpha );

/// Element-wise (Hadamard): `out[i] = a[i] * b[i]`.
cudaError_t cuda_mul_float( float const *a, float const *b, float *out, std::size_t count );
cudaError_t cuda_mul_float_inplace( float *a, float const *b, std::size_t count );
cudaError_t cuda_mul_fp8( __nv_fp8_e4m3 const *a, __nv_fp8_e4m3 const *b, __nv_fp8_e4m3 *out,
                          std::size_t count );
cudaError_t cuda_mul_fp8_inplace( __nv_fp8_e4m3 *a, __nv_fp8_e4m3 const *b, std::size_t count );

/// In-place scale: `dev[i] *= alpha`.
cudaError_t cuda_scale_float( float *dev, std::size_t count, float alpha );
cudaError_t cuda_scale_fp8( __nv_fp8_e4m3 *dev, std::size_t count, float alpha );

/// Element-wise unary: `out[i] = f(in[i])` (separate buffers).
cudaError_t cuda_cwise_greater_float( float const *in, float *out, std::size_t n, float scalar );
cudaError_t cuda_cwise_greater_fp8( __nv_fp8_e4m3 const *in, __nv_fp8_e4m3 *out, std::size_t n,
                                    float scalar );
cudaError_t cuda_cwise_one_minus_float( float const *in, float *out, std::size_t n );
cudaError_t cuda_cwise_one_minus_fp8( __nv_fp8_e4m3 const *in, __nv_fp8_e4m3 *out, std::size_t n );
cudaError_t cuda_cwise_sigmoid_float( float const *in, float *out, std::size_t n );
cudaError_t cuda_cwise_sigmoid_fp8( __nv_fp8_e4m3 const *in, __nv_fp8_e4m3 *out, std::size_t n );
cudaError_t cuda_cwise_exp_float( float const *in, float *out, std::size_t n );
cudaError_t cuda_cwise_exp_fp8( __nv_fp8_e4m3 const *in, __nv_fp8_e4m3 *out, std::size_t n );
cudaError_t cuda_cwise_log_float( float const *in, float *out, std::size_t n );
cudaError_t cuda_cwise_log_fp8( __nv_fp8_e4m3 const *in, __nv_fp8_e4m3 *out, std::size_t n );

cudaError_t cuda_sum_float( float const *in, std::size_t n, float *out_host );
cudaError_t cuda_sum_fp8( __nv_fp8_e4m3 const *in, std::size_t n, float *out_host );

cudaError_t cuda_sum_along_axis_float( float const *in, float *out, std::size_t rows, std::size_t cols,
                                       int axis, bool transpose_out );
cudaError_t cuda_sum_along_axis_fp8( __nv_fp8_e4m3 const *in, __nv_fp8_e4m3 *out, std::size_t rows,
                                     std::size_t cols, int axis, bool transpose_out );
cudaError_t cuda_max_along_axis_float( float const *in, float *out, std::size_t rows, std::size_t cols,
                                       int axis );
cudaError_t cuda_max_along_axis_fp8( __nv_fp8_e4m3 const *in, __nv_fp8_e4m3 *out, std::size_t rows,
                                     std::size_t cols, int axis );

cudaError_t cuda_argmax_along_axis_float( float const *in, unsigned int *out, std::size_t rows,
                                          std::size_t cols, int axis );
cudaError_t cuda_argmax_along_axis_fp8( __nv_fp8_e4m3 const *in, unsigned int *out, std::size_t rows,
                                        std::size_t cols, int axis );

/// Element-wise: `out[i] = in[i] + value`. In-place when `in == out`.
cudaError_t cuda_add_elementwise( float *in, float *out, std::size_t n, float value );
cudaError_t cuda_add_elementwise( __nv_fp8_e4m3 *in, __nv_fp8_e4m3 *out, std::size_t n,
                                  __nv_fp8_e4m3 value );

cudaError_t cuda_reduce_sum_to_dim( float *in, float *out, std::size_t prefix, std::size_t mid,
                                    std::size_t suffix );
cudaError_t cuda_reduce_sum_to_dim_axis_0( float *in, float *out, std::size_t n, std::size_t stride );
/// Fused multiply-subtract in-place: `data[i] -= other[i] * scalar`.
cudaError_t cuda_mul_substract_float( float *data, float const *other, std::size_t count,
                                      float scalar );

/// col2im scatter (backward of im2col for valid convolution).
cudaError_t cuda_col2im_float( float const *col, float *im, int N, int C_in, int H, int W,
                               int Kh, int Kw );

/// Batch-norm helpers (per-channel, \p channels == \f$C\f$). \p saved_inv_std is cuDNN \c resultSaveInvVariance
/// (\f$1/\sqrt{\mathrm{var}+\epsilon}\f$). Writes \p running_var as \f$1/\mathrm{saved}^2 - \epsilon\f$.
cudaError_t cuda_bn_running_var_from_saved_inv_std_float( float *running_var,
                                                          float const *saved_inv_std,
                                                          std::size_t channels, float eps );
/// Writes \p saved_inv as \c rsqrtf(var[i] + eps) (inverse std dev for cuDNN backward).
cudaError_t cuda_bn_saved_inv_std_from_variance_float( float *saved_inv, float const *var,
                                                       std::size_t channels, float eps );
} // namespace neural

#endif
