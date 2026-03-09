# Tensor API Specification

This document describes the **interface** your Tensor class should expose. Implement the types and methods yourself; the signatures below are a target API.

---

## 1. Data Type and Storage

- **Element type**: Use `float` (32-bit) for the main path. You can add a template `Tensor<T>` later if you want.
- **Dtype (optional)**: A simple enum or type alias (e.g. `float32`) is enough for now; full dtype dispatch can come later.
- **Storage**: Backed by Eigen (see [BACKEND.md](BACKEND.md)). Contiguous, row-major layout.

---

## 2. Shape and Dimensions

- **Rank**: For step 1, **2D (matrix)** is enough: rows and columns.
- **Shape**: Provide at least:
  - `rows()` → number of rows
  - `cols()` → number of columns
  - `size()` → total number of elements (rows × cols)
- Optional: `shape()` returning a small array or pair `(rows, cols)` for consistency with “tensor” terminology.

---

## 3. Construction and Creation

Implement at least:

- **Default constructor**: Optional; or require dimensions.
- **Constructor** `Tensor(rows, cols)`: allocate and optionally zero-initialize.
- **Constructor** from existing data (e.g. `Tensor(rows, cols, float* data)` or `std::vector<float>`) for tests and loading.
- **Zero / constant**: Either constructor or static helper, e.g. `Tensor::zeros(rows, cols)`, `Tensor::constant(rows, cols, value)`.
- **Random initialization**: Useful for weights; e.g. `Tensor::random(rows, cols)` with a simple uniform or normal distribution (can delegate to `<random>`).

---

## 4. Element Access

- **Single element**: `operator()(i, j)` or `at(i, j)` for the element at row `i`, column `j`. Bounds-check in debug if you like.
- **Data pointer**: `data()` returning `float*` (or `const float*`) for interoperability and debugging. Eigen backend gives this via `.data()`.

---

## 5. Operations Required for the Linear Layer and Loss

These are the operations used in forward/backward (see PLAN.md and THEORY.md).

| Operation | Description | Used in |
|-----------|-------------|--------|
| **Matmul** | C = A × B (matrix product). Shape: (M,K) × (K,N) → (M,N). | Linear: y = xW + b; Backward: dL/dW = xᵀ @ dL/dy, dL/dx = dL/dy @ Wᵀ |
| **Transpose** | Aᵀ: (rows, cols) → (cols, rows). | Backward formulas above |
| **Element-wise add** | C = A + B (same shape). | y = xW + b; adding bias |
| **Element-wise multiply** | C = A * B (same shape). | ReLU backward: grad_input = grad_output * mask |
| **Multiply by scalar** | B = α * A. | Gradient scaling |
| **Sum** | Sum over all elements, or over one axis. | dL/db = sum(dL/dy) over batch |

Suggested API forms (names can vary):

- `Tensor matmul(const Tensor& other) const;`
- `Tensor transpose() const;`
- `Tensor operator+(const Tensor& other) const;` and `operator+=`
- `Tensor operator*(const Tensor& other) const;` (element-wise); `Tensor operator*(float scalar) const;`
- `float sum() const;` (all elements). Optionally: `Tensor sum_along_axis(int axis) const;` if you need batched gradients.

---

## 6. Copy and Move

- **Copy constructor** and **copy assignment**: deep copy of data.
- **Move constructor** and **move assignment**: optional but recommended for returning tensors from functions and avoiding unnecessary copies.

---

## 7. Optional Extras (Later)

- **Broadcasting**: For (batch, features) + (features,) bias; you can start by requiring same shape and add broadcasting when you implement batching.
- **Reshape / view**: Change interpretation of dimensions without copying (advanced; Eigen blocks can help).
- **3D+ tensors**: For batches (batch, rows, cols) or (batch, features); add when you need them.

---

## 8. Testing Checklist (test_tensor.cpp)

- **Shape**: After construction and after operations, `rows()`, `cols()`, `size()` match expectations.
- **Matmul**: Multiply two known matrices, compare result to hand-computed or known reference.
- **Transpose**: (Aᵀ)ᵀ = A; (AB)ᵀ = Bᵀ Aᵀ.
- **Element-wise**: Add, multiply two tensors; multiply by scalar; check a few elements.
- **Sum**: `sum()` on a known tensor equals expected value.
- **Invalid input**: Optional: dimensions incompatible for matmul (e.g. (2,3) × (2,3)) — assert or throw.

Implement the types and method bodies yourself; use this document as the contract your implementation must satisfy.
