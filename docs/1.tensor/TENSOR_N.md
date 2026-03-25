# TensorN — Basic Operations (Specification)

This document lists the **core operations** intended for `neural::TensorN<Rank, T>`: a **fixed-rank** dense tensor (CPU-first), complementing the 2D `Tensor` used for fully connected layers.

Use it as a **design target** when implementing `include/tensors/tensor_n.hpp`; names and exact signatures can match your existing style (`Tensor`, Eigen usage in [BACKEND.md](BACKEND.md)).

---

## 1. Role and scope

- **Purpose**: Hold activations and parameters for **convolutional** and other **N-D** pipelines (e.g. batch × channels × height × width).
- **Element type**: Template `T`, default `float`, aligned with `Tensor<T>`.
- **Backend**: Same story as 2D tensors—**Eigen** on CPU for storage and many ops; optional CUDA later.

---

## 2. Rank, shape, and layout

| Concept | Intended behavior |
|--------|-------------------|
| **Rank** | Number of dimensions (e.g. 4 for NCHW). Exposed as `rank()` or `ndim()`. |
| **Shape** | Extents per axis, e.g. `{N, C, H, W}`. Return by value (`std::array` / `std::span` / small vector) or read-only view. |
| **Size** | Total element count = product of shape (0 if any extent is 0). |
| **Strides** | Bytes or elements per step along each axis; **row-major** contiguous storage is the default (last axis stride = 1). |
| **Contiguity** | Optional `is_contiguous()`; reshapes/views may require it. |

### Slice and reduction: result must be a smaller tensor

Operations that **restrict** which elements you keep, or **aggregate** along axes, must produce a value whose **shape and/or element count** reflects that—otherwise the API is misleading.

| Operation | Expected result |
|-----------|-----------------|
| **Slice** (proper sub-range on one or more axes) | A **new** tensor or view whose **extents** are ≤ the original; **smaller** `numel` unless every range is full. **Return type by rank:** if the slice is **2-dimensional** (rows × cols), return **`Tensor<T>`**—the same matrix type used by linear layers—not `TensorN<2, T>`. Rank **0** → scalar `T` (or rank-0 wrapper). Rank **1** or **≥ 3** → `TensorN` of that rank (or a documented 1D convention). |
| **Reduce along one axis** (e.g. `sum` over `c`) **without** `keepdim` | Result has **rank one less** than the input (that axis disappears). **Fewer** elements than the input. If the result is **2D**, return **`Tensor<T>`**; otherwise **`TensorN<Rank - 1, T>`** when rank is a template parameter. |
| **Reduce along one axis** **with** `keepdim` | Same **rank** as input, but the reduced axis has **extent 1**. Still **fewer** elements than the input unless that axis was already length 1. |
| **Reduce over all elements** | A **scalar** `T`, or a dedicated **0-D / rank-0** tensor type—pick one convention and use it consistently. Not a full-sized tensor with the original shape. |

**Views** (non-owning slices) still point at a **subset** of the logical index space: the **reported shape** of the view must match the smaller sub-tensor, not the parent’s shape.

### Why `Tensor` for 2D slices

The existing **`Tensor`** type is **2D** (Eigen matrix). Anything that is **naturally a matrix**—for example the **first slice** along batch when the remaining axes are **two** (e.g. shape `(N, M, K)` and you take index `0` → `(M, K)`)—should return **`Tensor<T>`** so call sites can pass it to **`Tensor`** matmul and FC layers without conversion.

If the first batch slice of a **4D** tensor `(N, C, H, W)` is **3D** `(C, H, W)`, that is **not** a matrix; return **`TensorN<3, T>`** (or offer a separate helper such as **`to_matrix()`** / reshape to `(C, H*W)` as **`Tensor`** when you explicitly want a matrix for a linear layer).

### Zero-copy 2D views: `Eigen::Map` vs owning `Tensor`

Your current **`Tensor`** holds an **`Eigen::Matrix<..., RowMajor>`**, which **owns** its storage. Building a **`Tensor`** from a slice **copies** into that matrix unless you add a separate **view** path.

**Eigen can alias existing memory without a copy:**

- **`Eigen::Map<MatrixType>`** — construct with a **`T*`** pointer, **`rows`**, **`cols`**. The map reads/writes the same bytes as the parent buffer (layout must match row-major **(i,j)** indexing up to stride).
- **Non-contiguous** 2D slabs (e.g. strided channel slice) use **`Eigen::Stride`** with the map: outer stride = jump between rows, inner stride = jump between columns (often `1` for dense rows). See [BACKEND.md](BACKEND.md) §6 and Eigen’s `Map` / `Stride` docs.

**Design options:**

| Approach | Copy? | Notes |
|----------|-------|--------|
| Return **`Tensor`** from slice | **Yes** (unless you change `Tensor`) | Simple ownership; safe if parent is dropped. |
| Return **`Eigen::Map<...>`** or a thin **`TensorView`** / **`TensorMap`** wrapper | **No** | Pointer + shape (+ stride); **parent `TensorN` must outlive** the view. |
| APIs take **`Eigen::Ref<const Matrix>`** | **No** when passing existing matrix/map | Good for kernels that should not allocate. |

So: **yes**, Eigen supports exactly “hold a pointer and treat it as a matrix” via **`Map`** (and **`Ref`** for parameters). Your **`Tensor`** type does **not** do that today; add a **dedicated view type** or an **optional** non-owning mode if you want slice → 2D **without** copy, and document **lifetime** (view borrows storage from the owning `TensorN`).

For this design pattern (no code in-repo), see [TENSOR_VIEW_REF.md](TENSOR_VIEW_REF.md).

---

## 3. Construction and factories

- Default construct (empty / rank 0 / deferred allocation—pick one convention and document it).
- Construct from **shape** (and optionally fill value): zeros, uninitialized, or constant.
- Construct from **existing buffer** + shape (+ strides if non-standard), with clear ownership rules (owning vs non-owning view).
- **Copy** / **move** constructors and assignment (deep copy on copy).
- Static helpers if useful: `zeros(shape)`, `ones(shape)`, `full(shape, value)`, optional `random_uniform` / `random_normal` for weights.

---

## 4. Element access

- **Multi-index** access: `operator()(i, j, ...)` or `at(i, j, ...)` with arity matching rank (or checked indexing with a span of indices).
- **Linear index**: `data()[k]` or `operator[](k)` when layout is known contiguous.
- **`data()` / `data() const`**: raw pointer for tests, I/O, and Eigen/tensor maps.

---

## 5. Views, reshape, and axis manipulation

- **Reshape**: Same total size, new shape (contiguous storage); error if incompatible.
- **View / slice**: Read-only or read-write **non-owning** view over a sub-region (ranges per axis), without copying when possible. The returned object’s **shape** is the **sliced** shape (smaller tensor), not the parent’s shape. **Rank-2** result → **`Tensor<T>`**; other ranks → **`TensorN`** (see §2).
- **Permute / transpose axes**: Reorder dimensions (e.g. NCHW → NHWC); may return a new tensor or a view depending on layout.
- **Squeeze / unsqueeze**: Remove or add length-1 dimensions.
- **Flatten**: Collapse to 1D or to 2D `(batch, rest)` for bridging to linear layers.

---

## 6. Element-wise operations

Same spirit as `Tensor`: require **broadcasting rules** to be defined (at minimum: same shape; later: scalar and aligned axis broadcast).

| Category | Examples |
|----------|----------|
| Arithmetic | `+`, `-`, `*` (element-wise), `/`, `+=`, `-=`, `*=`, `/=` |
| Unary | Negation, `exp`, `log`, `relu` / `max(0,x)` if you want them on the tensor type |
| Scalar | Multiply/divide/add by scalar |

In-place variants (`*=` etc.) reduce allocations in hot paths.

---

## 7. Reductions and aggregations

- **Sum / mean / max / min** over **all** elements → **scalar** (or rank-0 tensor); over a **single axis** → tensor with **smaller** `numel`, and **rank** reduced by one unless `keepdim` is true (see §2).
- **Argmax** along an axis (useful for classification metrics): result has **smaller** shape along that axis (typically length 1 with `keepdim`, or axis removed without it).

Reductions must **not** return a tensor of the **same** full shape as the input unless the operation is a no-op (e.g. sum over an axis of length 1 with keepdim, which is degenerate).

---

## 8. Linear algebra (as needed for CNNs)

- **Batch matrix multiply** when tensors are interpreted as stacks of 2D matrices (e.g. `(batch, M, K) × (batch, K, N)`).
- **Dot / contraction** along specified axes (general case; can start with 2D slices + existing `Tensor` matmul).

Exact API can defer to thin wrappers that extract 2D `Tensor` views from `TensorN` slices.

---

## 9. Interop with 2D `Tensor`

- **From `Tensor`**: Promote a matrix to `TensorN` with shape `(1, rows, cols)` or `(rows, cols)` as rank-2, as you standardize.
- **To `Tensor`**: Any slice or operation whose result is **2D** returns **`Tensor`** directly (§2). For `(batch, M, K)`, index one batch → `(M, K)` **`Tensor`**. For `(N, C, H, W)`, a single batch index alone yields **3D**; use **`TensorN<3>`** or an explicit **reshape** to `(C, H*W)` **`Tensor`** when you need a matrix.

---

## 10. Optional later

- **Padding / cropping** spatial axes (often done inside conv layers instead).
- **Explicit im2col / col2im** (may live on conv utilities rather than `TensorN`).
- **GPU buffer** view sharing with a CUDA type when you add a device backend.

---

## Example: two RGB images

A small batch of **two RGB images** is naturally a **4D** tensor. A common layout is **NCHW**: batch (**N**), color **channels** (**C** = 3 for R, G, B), **height** (**H**), **width** (**W**).

- **N = 2** → two images  
- **C = 3** → R, G, B planes  
- **H, W** → pixel grid (e.g. 224×224 for real inputs)

**Shape** is `(2, 3, H, W)`; **element count** is `2 × 3 × H × W`.

### Indexing (NCHW)

Element **`(n, c, y, x)`** is: image **`n`**, channel **`c`** (0 = R, 1 = G, 2 = B), row **`y`**, column **`x`**.

With **row-major** storage and NCHW, the **last** index **`x`** changes fastest along memory. Typical order of planes in a contiguous buffer:

1. Image 0, all of **R**, then **G**, then **B**  
2. Image 1, **R**, then **G**, then **B**

So you “assign” pixels by filling **`tensor(n, c, y, x)`** (or equivalent) for each image, channel, and spatial position—or by loading a **flat buffer** that already follows this order and wrapping it with shape `(2, 3, H, W)`.

### Tiny numeric example (H = 1, W = 2)

Shape **`(2, 3, 1, 2)`** — two images, three channels, 1×2 pixels:

| Image | Channel | Row values `(x=0, x=1)` |
|-------|---------|-------------------------|
| 0 | R | 1, 2 |
| 0 | G | 3, 4 |
| 0 | B | 5, 6 |
| 1 | R | 7, 8 |
| 1 | G | 9, 10 |
| 1 | B | 11, 12 |

**Linear order** (row-major NCHW, flattening):  
`1, 2` (img0 R) → `3, 4` (img0 G) → `5, 6` (img0 B) → `7, 8` (img1 R) → `9, 10` (img1 G) → `11, 12` (img1 B).

For conv layers you usually keep this **4D** block as-is. A literal **2D matrix** view appears only after **reshape** or helpers like **im2col**, not by treating the whole batch as one arbitrary rows×cols matrix without defining layout.

### Alternative: NHWC

Some stacks use **NHWC**: shape **`(2, H, W, 3)`**. At each **`(n, y, x)`** the three numbers **R, G, B** are adjacent in memory. The same pixel data can be represented in either convention; **permute** swaps NCHW ↔ NHWC when interfacing with code that expects the other order.

---

## Summary checklist

Minimal useful surface for CNN prototyping:

1. Rank, shape, strides, size, contiguous flag  
2. Allocate / fill / copy / move  
3. Multi-index and linear `data()` access  
4. Reshape, slice/view, axis permute, flatten  
5. Element-wise math + reductions over all or one axis  
6. Clear story for `Tensor` ↔ `TensorN` conversion  

Implement in whatever order your layers need first (often: shape + data + reshape + element-wise + sum over batch/channel axes).
