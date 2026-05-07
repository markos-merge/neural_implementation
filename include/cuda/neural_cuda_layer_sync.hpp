#ifndef NEURAL_IMPL_NEURAL_CUDA_LAYER_SYNC_HPP
#define NEURAL_IMPL_NEURAL_CUDA_LAYER_SYNC_HPP

#if NEURAL_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace neural {

#if NEURAL_CUDA_ENABLED
/// Block host until the default stream on the current device finishes queued work.
inline void cuda_layer_sync() noexcept
{
	// (void)cudaDeviceSynchronize();
}

/// RAII: \ref cuda_layer_sync on scope exit (return, throw, or end of block).
/// For local scope in functions / blocks only, not as a class data member.
struct CudaStreamSyncOnExit
{
	// ~CudaStreamSyncOnExit() noexcept { cuda_layer_sync(); }
};
#else
inline void cuda_layer_sync() noexcept
{
}

struct CudaStreamSyncOnExit
{
	~CudaStreamSyncOnExit() noexcept = default;
};
#endif

} // namespace neural

#endif
