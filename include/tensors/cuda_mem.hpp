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

} // namespace detail
} // namespace neural

#endif // NEURAL_CUDA_ENABLED

#endif // NEURAL_IMPL_CUDA_MEM_HPP
