#ifndef NEURAL_IMPL_NEURAL_CUDA_LAYER_SYNC_HPP
#define NEURAL_IMPL_NEURAL_CUDA_LAYER_SYNC_HPP

#if NEURAL_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace neural {

#if NEURAL_CUDA_ENABLED
/// Block host until the default stream on the current device finishes queued work.
inline void cuda_layer_sync()
{
	(void)cudaDeviceSynchronize();
}
#else
inline void cuda_layer_sync()
{
}
#endif

} // namespace neural

#endif
