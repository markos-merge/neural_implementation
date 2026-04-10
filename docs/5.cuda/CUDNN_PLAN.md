# cuDNN implementation plan

This document tracks everything that needs to be implemented or refactored to bring the cuDNN-backed `CudaTensorN` GPU path to feature-parity with the CPU `TensorN` path.

---

## Already done (this session)

| File | What |
|------|------|
| `CMakeLists.txt` | `find_path` / `find_library` for cuDNN, `NEURAL_CUDNN_ENABLED` define, `cudnn_handle.cpp` source, links `${CUDNN_LIBRARY}` |
| `include/cuda/cudnn_handle.hpp` | `CudnnHandle` singleton matching the `CublasHandle` pattern |
| `src/cuda/cudnn_handle.cpp` | Lazy `cudnnCreate` in `instance()`, `cudnnDestroy` in destructor |
| `include/cuda/neural_cuda_kernels.hpp` | Declarations for `cuda_mul_substract_float` and `cuda_col2im_float` |
| `src/cuda/cuda_runtime.cu` | Kernel and wrapper implementations for the two above |

---

## Step 1 — `CudaTensorN` header  (`include/tensors/cuda_tensor_n.hpp`)

The entire `CudaTensorN<rank, T>` class. When `NEURAL_CUDNN_ENABLED=0` the header aliases it to `TensorN<rank, T>`.

### 1.1 Memory management

| Item | Notes |
|------|-------|
| `T* m_data` device pointer | `cudaMalloc` / `cudaFree` |
| Default constructor | `m_data = nullptr`, zero shape/strides |
| Shape constructor | allocate + zero-fill via `cuda_fill_float` |
| Value constructor | allocate + fill via `cuda_fill_float` |
| Range constructor `(shape, begin, end)` | `cudaMemcpy(..., cudaMemcpyDefault)` — works for both host and device source pointers via UVA |
| `(vector, shape)` constructor | host→device copy |
| Copy constructor / copy-assign | `cudaMemcpyDeviceToDevice` |
| Move constructor / move-assign | steal pointer, null the source |
| Destructor | `cudaFree` |
| `tensor_alias<other_rank, type>` | `= CudaTensorN<other_rank, type>` |

### 1.2 Basic accessors

| Method | Notes |
|--------|-------|
| `data()` | returns device pointer `T*` |
| `shape()` | host-side `m_shape` |
| `strides()` | host-side `m_strides` |
| `size()` | product of `m_shape` |
| `operator()(indices)` | `cudaMemcpy` single element D→H (slow, debug only) |
| `at(n,c,h,w)` `requires(rank==4)` | same, single element D→H |

### 1.3 Assign / mutation helpers

| Method | Implementation |
|--------|---------------|
| `reshape(new_shape)` | update `m_shape` / `m_strides` in-place, no data move |
| `assign(vector)` | `cudaMemcpy` H→D |
| `assign(ptr, size)` | `cudaMemcpy(..., cudaMemcpyDefault)` |
| `assign(indices, value)` | `cudaMemcpy` single element H→D |
| `addElementwise(indices, value)` | read element D→H, add, write H→D (slow, infrequent) |
| `assign(from_slice, to_slice, tensor)` | see refactor note § 3.1 |
| `randomizeHe(fan_in)` | generate on host, bulk H→D copy |

### 1.4 Element-wise operations (reuse existing kernels)

| Method | Kernel |
|--------|--------|
| `operator*=(TensorN const&)` | `cuda_mul_float_inplace` |
| `operator*=(scalar)` | `cuda_scale_float` |
| `cwiseGreaterInPlace(other, scalar)` | `cuda_cwise_greater_float` |
| `mulNSubstractInPlace(other, scalar)` | `cuda_mul_substract_float` ✅ (new) |

### 1.5 cuDNN / cuBLAS operations

| Method | Library call | Notes |
|--------|-------------|-------|
| `im2ColConvolution(kernel, im2col_ws, output, gemm_ws)` | `cudnnConvolutionForward` | `pad=0` (valid conv); `im2col_ws` and `gemm_ws` params ignored — workspace allocated internally per-call (optimise later, see § 3.3) |
| `addColorChannelInPlace(bias)` `requires(rank==4)` | `cudnnAddTensor` | broadcast `{1,C,1,1}` bias over `{N,C,H,W}` |
| `reduceSumToDim(axis=0, output_rank4)` `requires(rank==4)` | `cublasSgemv` with an internal ones-vector | `output` gets shape `{1, shape[0], 1, 1}` |
| `reduceSumToDim(axis=0, output_rank1)` | same as above | `output` gets shape `{shape[0]}` |
| `swapAxes(new_axes, output)` | `cudnnTransformTensor` with computed per-axis strides | works for any rank-4 permutation |
| `multiply(tensor_2d, transpose_second, output)` `requires(rank==4)` | `cublasSgemm` | `this` treated as `(shape[0], size/shape[0])` matrix |
| `col2Im(kernel_shape, orig_shape, output)` `requires(rank==4)` | `cuda_col2im_float` ✅ (new kernel) | valid convolution inverse scatter |
| `im2Col(kernel_shape, output)` | not needed for cuDNN forward — throw or leave as stub | only present for API completeness |

### 1.6 RAII descriptor helpers (detail namespace, header-internal)

Small structs with constructor/destructor for:
- `CudnnTensorDesc` → `cudnnCreateTensorDescriptor` / `cudnnDestroyTensorDescriptor`
- `CudnnFilterDesc` → `cudnnCreateFilterDescriptor` / `cudnnDestroyFilterDescriptor`
- `CudnnConvDesc` → `cudnnCreateConvolutionDescriptor` / `cudnnDestroyConvolutionDescriptor`
- `CudnnReduceDesc` → `cudnnCreateReduceTensorDescriptor` / `cudnnDestroyReduceTensorDescriptor`

---

## Step 2 — Layers that need a cuDNN-aware backward

The `ConvolutionalLayer<CudaTensorN>` **forward** works once step 1 is done.  
The **backward** is broken because it relies on `m_im2col_aux_tensor` being populated by an explicit `im2Col` call during forward — which the cuDNN path skips.

### 2.1 `ConvolutionalLayer` backward refactor

Two options:

**Option A — Partial compatibility (quickest)**  
During `im2ColConvolution`, also call `im2Col` to populate `m_im2col_aux_tensor` as a side-effect.  
The cuDNN forward result is correct; backward reuses the col matrix via the existing GEMM+col2Im path.  
*Downside*: im2Col kernel runs unnecessarily on the GPU (needs a custom im2col CUDA kernel), no speedup for backward.

**Option B — Full cuDNN backward (recommended)**  
Add a template specialisation (or a virtual/CRTP hook) in `ConvolutionalLayer` for the cuDNN path:
- Weight gradient: `cudnnConvolutionBackwardFilter(dy, x → dW)`
- Input gradient: `cudnnConvolutionBackwardData(dy, W → dx)`
- Bias gradient: `cudnnConvolutionBackwardBias(dy → db)` (replaces `reduceSumToDim`)

This removes `swapAxes`, `multiply`, `col2Im` from the backward hot path entirely.

### 2.2 `MaxPoolLayer` backward

Current backward uses `TensorN` loops. For `CudaTensorN`:
- Forward: `cudnnPoolingForward`
- Backward: `cudnnPoolingBackward`

Needs its own pool descriptor with `CUDNN_POOLING_MAX`.

### 2.3 Activation layers

| Layer | cuDNN path |
|-------|-----------|
| `ReLULayer` | `cudnnActivationForward/Backward` with `CUDNN_ACTIVATION_RELU` |
| `SigmoidLayer` | `cudnnActivationForward/Backward` with `CUDNN_ACTIVATION_SIGMOID` |
| `SoftmaxLayer` | `cudnnSoftmaxForward/Backward` with `CUDNN_SOFTMAX_ACCURATE` |

Rank-4 (NCHW) tensors need 4D descriptors here too.

### 2.4 `BatchNormLayer`

- Forward: `cudnnBatchNormalizationForwardTraining` / `ForwardInference`
- Backward: `cudnnBatchNormalizationBackward`

Needs scale/bias/mean/variance device tensors.

---

## Step 3 — Required refactors

### 3.1 `assign(from_slice, to_slice, tensor)` overload

The templated slice-assign in `TensorN` copies a sub-block between tensors of different ranks. The `CudaTensorN` equivalent needs a kernel or `cudaMemcpy2D`. Currently not exercised by `ConvolutionalLayer`; implement when needed.

### 3.2 `CMakePresets.json` — cuDNN preset

Add `release-cudnn` and `relwithdebinfo-cudnn` presets that set `NEURAL_ENABLE_CUDA=ON` plus `CUDNN_ROOT=<path>` as a cache variable so CI and developers can opt in cleanly.

### 3.3 Workspace caching

Allocating and freeing a cuDNN workspace on every `im2ColConvolution` call is expensive. Refactor options:
- **Per-tensor cache**: add `void* m_workspace; size_t m_workspace_bytes` to `CudaTensorN` and resize lazily.
- **Global workspace pool**: a singleton `CudnnWorkspace` that holds the largest workspace seen so far and reallocates only when it grows.

### 3.4 cuDNN algorithm selection

`cudnnGetConvolutionForwardAlgorithm_v7` is a heuristic (no benchmarking). Replace with `cudnnFindConvolutionForwardAlgorithmEx` (benchmarks real times) and cache the chosen `cudnnConvolutionFwdAlgo_t` per `(input_shape, kernel_shape)` key. Same for backward filter and backward data.

### 3.5 Training loop host↔device transfer

The CIFAR/MNIST loaders produce CPU-side `Tensor`/`TensorN` objects. A host→device copy is needed before each forward pass. This should happen in a dedicated transfer step (or a `CudaTensorN` constructor from `TensorN`) rather than inside the layer.

### 3.6 `im2Col` on device (for Option A backward)

If Option A is chosen for step 2.1, a `cuda_im2col_float` kernel is needed (inverse of `cuda_col2im_float`) so `im2ColConvolution` can populate the col workspace on device during the forward pass.

---

## Step 4 — Testing

| Test | What to check |
|------|--------------|
| `test_cuda_tensor_n.cpp` (new) | basic alloc, copy, reshape, swapAxes round-trip |
| Forward convolution parity | `CudaTensorN` forward vs `TensorN` forward on same weights/input |
| Backward parity | weight and input gradients agree with CPU reference |
| Bias add | `addColorChannelInPlace` matches CPU broadcast |
| Reduce sum | `reduceSumToDim` matches CPU |
| Optimizer step | `mulNSubstractInPlace` matches CPU SGD update |

---

## Dependency order

```
Step 1 (CudaTensorN memory + accessors)
  └─► Step 1 (element-wise kernels)
        └─► Step 1 (im2ColConvolution forward via cuDNN)
              └─► Step 2.1 backward (Option A or B)
                    └─► Step 2.2 MaxPool
                          └─► Step 2.3 Activations
                                └─► Step 2.4 BatchNorm
                                      └─► Step 3 refactors (workspace, algo cache)
                                            └─► Step 4 tests
```

Refactors 3.2 (presets) and 3.5 (data transfer) can happen independently at any time.
