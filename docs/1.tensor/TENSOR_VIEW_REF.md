# Tensor, TensorView, and Eigen::Ref — design note (no implementation here)

You do **not** need to reimplement every `Tensor` operation for a view type. Eigen already implements dense linear algebra on **expressions** that share a common interface.

## Core idea

1. **`Tensor`** keeps an owning **`Eigen::Matrix<..., RowMajor>`**.
2. A **view** type can hold an **`Eigen::Map<`**same matrix type**>`** over borrowed storage (zero-copy).
3. **`Eigen::Ref<const MatrixType>`** (and mutable `Ref`) can bind to **both** a `Matrix` and a `Map` without copying storage.

So helpers written in terms of **`Ref<const matrix_type>`** can run the **same** Eigen code for owned and viewed data.

## How to extend without duplicating logic

For each operation you care about:

1. Implement **one** function that takes **`Eigen::Ref<const matrix_type>`** (or mutable `Ref` where in-place writes are allowed).
2. Have **`Tensor::foo`** forward storage as **`Ref`** to that helper.
3. Have the **view** type’s **`foo`** forward **`Ref`** from its **`Map`** the same way.

Eigen’s element-wise ops, reductions, `transpose`, etc. work on **`Ref`** / **`Map`** the same way as on **`Matrix`**.

## Result type

Many ops naturally produce a **new** dense matrix (e.g. matrix product). Those can still return **`Tensor`** (owning result). The **inputs** can be views; only the **output** allocates.

## Lifetime

A view must not outlive the buffer it maps. **`Ref`** inside a function call is only valid for that call—typical Eigen usage.

## Strided / non-contiguous slices

A contiguous row-major `(rows, cols)` window uses a plain **`Map`**. Arbitrary strided 2D windows need **`Eigen::Stride`** on the **`Map`**; the same **`Ref`**-based pattern applies once the **`Map`** is set up correctly.
