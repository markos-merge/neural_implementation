#ifndef NEURAL_IMPL_CUDA_MEM_HPP
#define NEURAL_IMPL_CUDA_MEM_HPP

#if NEURAL_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstddef>
#include <chrono>
#include <thread>

namespace neural {
namespace detail {

inline cudaError_t cuda_malloc_retry( void **dev_ptr, std::size_t size_bytes,
                                      int max_attempts = 5 )
{
	for ( int attempt = 0; attempt < max_attempts; ++attempt ) {
		if ( attempt > 0 ) {
			std::this_thread::sleep_for( std::chrono::milliseconds( 1 * attempt ) );
			(void)cudaDeviceSynchronize();
		}
		cudaError_t const err = cudaMalloc( dev_ptr, size_bytes );
		if ( err == cudaSuccess ) {
			return cudaSuccess;
		}
		if ( err != cudaErrorMemoryAllocation ) {
			return err;
		}
	}
	return cudaErrorMemoryAllocation;
}

/// Reuse a single device buffer for cuDNN convolution workspaces: grows with \p need, never shrinks.
/// \p *ptr / \p *capacity_bytes updated on successful (re)allocation; safe to call with \p need == 0 (no buffer required).
inline cudaError_t cudnn_workspace_ensure( void **ptr, std::size_t *capacity_bytes, std::size_t need )
{
	if ( ptr == nullptr || capacity_bytes == nullptr ) {
		return cudaErrorInvalidValue;
	}
	if ( need <= *capacity_bytes ) {
		return cudaSuccess;
	}
	if ( *ptr != nullptr ) {
		(void)cudaFree( *ptr );
		*ptr = nullptr;
		*capacity_bytes = 0;
	}
	if ( need == 0 ) {
		return cudaSuccess;
	}
	cudaError_t const err = cuda_malloc_retry( ptr, need );
	if ( err == cudaSuccess ) {
		*capacity_bytes = need;
	}
	return err;
}

} // namespace detail
} // namespace neural

#endif // NEURAL_CUDA_ENABLED

#endif // NEURAL_IMPL_CUDA_MEM_HPP
