# Tensor Implementation Details

This folder contains implementation guidance for the **Tensor/Matrix** component—the first step in the implementation order (see [PLAN.md](../PLAN.md)).

## What You Are Building

A **Tensor** is the core data structure for the neural network: it holds multi-dimensional arrays of numbers (scalars, vectors, matrices, or higher-dimensional arrays) with a well-defined **shape** and **data type**. All layer inputs/outputs, weights, and gradients will be tensors.

## Scope for Step 1

- **CPU backend only** (Eigen). GPU abstraction comes later.
- Focus on **2D tensors (matrices)** first; you can extend to arbitrary rank later if needed for batches or convolutions.
- Implement the operations required by the **linear layer** and **loss** backprop: matrix multiply, transpose, element-wise ops, sum, and shape/creation utilities.

## Documents in This Folder

| File | Purpose |
|------|--------|
| [API.md](API.md) | Tensor interface: shape, dtype, constructors, and operations to implement. |
| [BACKEND.md](BACKEND.md) | Using Eigen as the CPU backend: storage, row-major layout, and mapping to Eigen types. |

## Implementation Order (Suggested)

1. Define the **Tensor** class: storage (Eigen matrix), shape, and dtype.
2. Implement **constructors** and **shape** accessors (e.g. `rows()`, `cols()`, `size()`).
3. Implement **element-wise** operations: add, subtract, multiply by scalar, and (if needed) multiply/divide element-wise with another tensor.
4. Implement **transpose** and **matrix multiply** (matmul).
5. Implement **sum** (over all elements or over an axis) for gradient accumulation (e.g. `dL/db`).
6. Add **tests** in `tests/test_tensor.cpp` (Catch2) for shape, matmul result, transpose, and element-wise ops.

## Success Criteria (from PLAN.md)

- `test_tensor.cpp` verifies: matmul result, transpose, shape, element-wise ops.
- The Tensor type can be used as input/output and weights in the linear layer (next step).

## Learning-First Reminder

Use these docs as a **spec and guide**. Implement the types and functions yourself; refer to [BACKEND.md](BACKEND.md) for Eigen usage hints and [API.md](API.md) for the intended interface.
