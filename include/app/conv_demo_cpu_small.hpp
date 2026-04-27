#ifndef CONV_DEMO_CPU_SMALL_HPP
#define CONV_DEMO_CPU_SMALL_HPP

/// CIFAR-10 "small" conv net on host tensors (im2col + Eigen), roughly aligned with
/// `run_conv_demo_cuda_small` in hyperparameters, but the trunk uses valid 3×3 (spatial shrinks)
/// and 1×1 convs at 2×2 to avoid invalid 3×3 on tiny maps. See the .cpp for details.
void run_conv_demo_cpu_small();

#endif
