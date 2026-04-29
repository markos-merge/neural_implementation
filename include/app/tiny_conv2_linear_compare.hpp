#ifndef TINY_CONV2_LINEAR_COMPARE_HPP
#define TINY_CONV2_LINEAR_COMPARE_HPP

#include <filesystem>

/// Tiny net: Conv→ReLU→Conv→ReLU→Linear (cuDNN “same” 3×3). Writes \p artifact_dir/*.f32 then prints a short summary.
/// Requires CUDA+cuDNN build. See \c scripts/tiny_conv2_linear_compare_pytorch.py for the PyTorch side.
void run_tiny_conv2_linear_compare( std::filesystem::path const &artifact_dir );

#endif
