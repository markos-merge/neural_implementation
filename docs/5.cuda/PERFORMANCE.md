# GPU performance: slow paths and what to change

When the CUDA path is correct but **much slower than expected**, the usual causes in this codebase are **per-element host/device traffic**, **full-device synchronization after many ops**, and **batch assembly that treats the GPU like a scalar array**. Below is an ordered list of components to address.

---

## 1. `SGDOptimizer` — batching (`include/sgd_optimizer.hpp`)

**Likely the largest training cost.**

- Each batch allocates new `batch_inputs` / `batch_targets` `CudaTensor`s (new device allocations every iteration).
- Batch assembly uses nested loops with `inputs[...](row, col)` and `batch_inputs.assign(...)`.
- `CudaTensor::operator()` performs one **device→host** `cudaMemcpy` per scalar read.
- `CudaTensor::assign` performs one **host→device** `cudaMemcpy` per scalar write.

So each batch element is touched with **two tiny transfers** instead of one contiguous upload.

**What to change:** Build batches on the **CPU** (or in a pinned host buffer), then **one** `cudaMemcpy` (or `cudaMemcpyAsync`) into **reused** device tensors; or store the full dataset in a layout that allows **slice/gather** on device without per-element copies.

---

## 2. `CudaTensor` scalar access (`include/cuda_tensor.hpp`)

`operator()` and `assign` are implemented with synchronous **single-element** `cudaMemcpy`. Any hot path that indexes tensors element-wise on the host will dominate runtime.

**What to change:** Avoid scalar access in loops; add batched APIs (e.g. upload from a contiguous host buffer, copy rows/slices, or a small gather kernel for shuffled indices).

---

## 3. `cudaDeviceSynchronize` after float add/subtract (`include/cuda_tensor.hpp`)

For float, `operator+`, `operator+=`, `operator-`, and `-=` call `cublasSgeam` then **`cudaDeviceSynchronize()`**. That **serializes the entire GPU** after every tensor add/subtract, which is heavier than the math itself when chained.

**What to change:** Drop redundant global syncs where stream ordering is enough; consider **CUDA streams** and a single sync per training step when needed for correctness or timing.

---

## 4. Synchronization inside `src/cuda_runtime.cu`

Many helpers (fills, reductions, `cuda_max_abs_*`, etc.) end with **`cudaDeviceSynchronize()`**. This stacks with the syncs in `CudaTensor` and inflates host-side wait time.

**What to change:** Prefer async launches and **one** sync per logical operation; reuse device scratch for reductions instead of allocating per call where it matters.

---

## 5. CIFAR demo evaluation (`src/cifar_demo.cpp`)

`argmax` and label inspection use `t(0, c)` in loops — each call is a **device→host** copy of one float.

**What to change:** Copy the full output row (or batch) to host once, or add a **device argmax** kernel and read back a single index.

---

## 6. `SoftmaxCrossEntropyLoss` (`include/cross_entropy_softmax_loss.hpp`)

Forward chains many separate operations (max along axis, subtract, exp, sum, normalize, log, multiply, sum). Each step allocates intermediates and launches kernels. Backward uses tensor subtraction, which (for float) hits the **syncing** add/subtract path in `CudaTensor`.

**What to change:** A **fused** softmax + cross-entropy forward (and matching backward) on device, with fewer temporaries; or a library path (e.g. cuDNN) if you adopt it.

---

## 7. `LinearLayer` backward (`include/linear_layer.hpp`)

`m_input.transpose().matmul(grad_output)` materializes a **transposed copy** of `m_input` for `CudaTensor` (copy constructor allocates and copies on device) every backward step.

**What to change:** Express transpose **inside** `matmul` via cuBLAS `OP_T` without allocating a separate transposed matrix, or add a strided/view + GEMM API.

---

## 8. Dataset layout (`include/cifar10_loader.hpp`)

The loader stores **one `CudaTensor` per sample** (e.g. `(1, 3072)`), so the dataset is many small allocations and non-coalesced access patterns when assembling batches the naive way.

**What to change:** Optional **single large matrix** `(N, features)` on device or host; batching then copies **contiguous** ranges or uses a gather kernel.

---

## 9. Roadmap alignment (`docs/5.cuda/PLAN.md`)

The existing plan already mentions **async copies**, **streams**, **kernel fusion**, and **device-side optimizer** — these match the gaps above.

---

## Suggested priority

1. Fix **SGDOptimizer batching** and **reuse** device batch buffers.
2. Remove or narrow **`cudaDeviceSynchronize`** in `CudaTensor` float add/subtract and in `cuda_runtime.cu` helpers.
3. Replace **scalar** `operator()` / `assign` in eval and any remaining hot paths.
4. **Fuse** softmax + CE and avoid **materialized transpose** in linear backward.
5. Optional: **larger contiguous** dataset storage and profiling (Nsight Systems / Compute) to validate.
