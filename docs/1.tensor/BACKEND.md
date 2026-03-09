# Tensor CPU Backend: Eigen

The plan specifies **Eigen** for the CPU backend. This document gives implementation hints only—you still write the code.

---

## 1. Why Eigen

- Header-only or easy to link; no separate “tensor” library needed for step 1.
- Provides dense matrices, matrix multiply, transpose, element-wise ops, and sum.
- Row-major layout by default; good for C++ and for interfacing with raw pointers.

---

## 2. Including Eigen

- Add Eigen to the project (e.g. FetchContent, system install, or submodule). Ensure `#include <Eigen/Dense>` (or the minimal headers you need) is available.
- For a 2D float matrix, the natural type is:
  - `Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>`
  - or the short alias: `Eigen::MatrixXf` — but note Eigen’s default is **column-major**. For row-major use the type above or `Eigen::Matrix<float, Dynamic, Dynamic, RowMajor>`.

---

## 3. Layout: Row-Major vs Column-Major

- **Row-major**: element at (i, j) is at index `i * cols + j`. C/C++ and many neural net codebases use row-major.
- **Column-major**: Eigen’s default; (i, j) is at `j * rows + i`.
- **Recommendation**: Use **row-major** for your Tensor so that `data()` and indexing match the usual “first index varies slowest” convention. Specify `Eigen::RowMajor` as the storage order in the matrix type.

---

## 4. Mapping Your Tensor to Eigen

- **Storage**: Your `Tensor` class can hold a single member of type `Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>` (or a type alias).
- **Shape**: `rows()` → `.rows()`, `cols()` → `.cols()`, `size()` → `.size()`.
- **Data pointer**: `.data()` returns a pointer to the first element; use for `data()` in your API.
- **Element access**: `operator()(i, j)` can delegate to the Eigen matrix `(i, j)`.

---

## 5. Implementing Operations (Hints)

- **Matmul**: Eigen uses `A * B` for matrix product. Check dimensions: (M,K) * (K,N) → (M,N). Eigen will assert on mismatch in debug.
- **Transpose**: `.transpose()` returns an expression; for a new tensor you need to evaluate into a dense matrix, e.g. `.transpose().eval()` (or assign to a new Eigen matrix), then wrap in your Tensor.
- **Element-wise**: Eigen’s `c = a.array() * b.array()` for element-wise product; `.array() + scalar` for adding scalar. Result back to matrix: `.matrix()`.
- **Sum**: `.sum()` over the whole matrix; for axis-wise sum you can use `.colwise().sum()` or `.rowwise().sum()` and then wrap the result.

---

## 6. Construction from Raw Data

- Eigen has `Eigen::Map`: you can wrap a `float*` and dimensions to get a matrix view (same layout). Useful for “construct from pointer” without copying if you guarantee lifetime; otherwise copy into an `Eigen::Matrix<>` and store that.

---

## 7. Build System

- In `CMakeLists.txt`, make Eigen available to the library target (e.g. `target_include_directories(neural_impl_lib PRIVATE ${EIGEN_INCLUDE_DIR})` or FetchContent). Do not link tests or bindings to the main executable; they link to `neural_impl_lib` which uses Eigen.

---

You implement the actual `Tensor` class and its methods; use these hints to integrate Eigen correctly and avoid layout or dimension bugs.
