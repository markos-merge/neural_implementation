# CUDA implementation plan

This document lays out the roadmap for GPU support alongside the existing CPU `Tensor` / Eigen path. Numbered topic folders under `docs/` follow the same pattern as `1.tensor`, `2.linear_layer`, `3.activation`, `4.optimizers`; CUDA work is tracked here under **`docs/5.cuda/`**.

---

## Already in place

- **CMake + CUDA**: optional CUDA language, `CUDA::cudart`, `CMAKE_CUDA_ARCHITECTURES` (e.g. `native`), `NEURAL_CUDA_ENABLED` for conditional compilation.
- **Runtime probe**: `neural::cuda_device_count()` in `neural_cuda_runtime.hpp` / `cuda_runtime.cu`.
- **`CudaTensor`**: public API declared; implementations are still stubs until device logic is added.
- **CPU reference**: `Tensor` + tests; `tests/test_tensor_helpers.hpp` for tolerant float comparisons (CPU vs GPU checks later).
- **Templated layers**: e.g. `LinearLayer<Tensor>` — the same templates can target `CudaTensor` once the backend implements the required operations.

---

## Phase 1 — Device memory and data movement

- Give **`CudaTensor`** real storage: device pointer, `rows` / `cols`, **row-major** layout (match CPU `Tensor`).
- **Allocate / free** (`cudaMalloc` / `cudaFree`, or a small allocator).
- **Host ↔ device**: `cudaMemcpy` and synchronization; later **`cudaMemcpyAsync`** and **CUDA streams** for overlap.
- **Data loaders** stay on CPU; each training step copies **batches** to device tensors.

---

## Phase 2 — Libraries and kernels

| Operation | Typical GPU approach |
|-----------|----------------------|
| **Matrix multiply** (`matmul`) | **cuBLAS** (`cublasSgemm`; later `cuBLASLt` or mixed precision) |
| **Bias / column-wise broadcast** | Match CPU semantics (`addColwise` = row-wise addition of a bias row); small kernels or fusion with GEMM |
| **Element-wise** (ReLU mask, sigmoid, etc.) | CUDA **kernels** or **Thrust** |
| **Reductions** (`sum`, `sum_along_axis`, `max_along_axis`) | Custom reduction kernels or **CUB** / library helpers |

---

## Phase 3 — Network on GPU

- Implement **`CudaTensor`** operations used by **linear** layers, **activations**, **softmax**, **loss**, and the **optimizer** (or keep the first optimizer version on CPU with explicit sync — not ideal long-term).
- Reuse **`LinearLayer<CudaTensor>`** (and similar) once `CudaTensor` implements the needed surface, **or** relax / split **`TensorLike`** so the GPU type does not have to mirror every CPU method unnecessarily.

---

## Phase 4 — Correctness and performance

- **Golden tests**: same shapes and seeds as CPU; compare outputs with **`require_tensor_close`** (after copying GPU results to host if needed).
- **Gradients**: finite-difference checks or small-network comparison against CPU backward.
- **Profiling**: Nsight Systems / Nsight Compute; then tune **streams**, **Tensor Cores** (FP16/BF16), and **kernel fusion**.

---

## Suggested implementation order

1. **`CudaTensor`**: allocate, free, host copy from/to `Tensor` or raw buffer; **`matmul`** via cuBLAS.
2. **Single linear forward** on synthetic data vs CPU `Tensor`.
3. **Linear backward** with cuBLAS.
4. **Activations and loss** on device.
5. **End-to-end training loop** (CPU load, GPU compute).
6. **Optimizer** on device parameters.

---

## Practical notes

- Keep **CPU `Tensor`** for reference math, debugging, and **dataset I/O**.
- Consider a **`CudaContext`** (or similar) holding **`cublasHandle_t`**, default **stream**, and error checking — either owned by `CudaTensor` or shared.
- If **`TensorLike`** is too strict for `CudaTensor`, introduce a slimmer concept for “what layers need” before forcing full parity with `Tensor`.

---

## Related files (codebase)

| Area | Location |
|------|----------|
| Stub GPU tensor API | `include/cuda_tensor.hpp` |
| CUDA CMake / `cudart` | `CMakeLists.txt` |
| Device count helper | `include/neural_cuda_runtime.hpp`, `src/cuda_runtime.cu` |
| Float comparison helpers for tests | `tests/test_tensor_helpers.hpp` |
