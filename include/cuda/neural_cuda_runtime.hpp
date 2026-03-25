#ifndef NEURAL_IMPL_NEURAL_CUDA_RUNTIME_HPP
#define NEURAL_IMPL_NEURAL_CUDA_RUNTIME_HPP

namespace neural {

/// Returns the number of CUDA devices, or -1 if CUDA is not built in or the runtime call fails.
int cuda_device_count();

/// True if at least one CUDA device is visible and a small device allocation succeeds.
/// Prefer this over \ref cuda_device_count alone when running kernels or allocating tensors.
bool cuda_runtime_ready();

} // namespace neural

#endif
