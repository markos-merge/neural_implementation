# Linear Layer — Theory

This document covers what you need to know to implement a **linear layer** (fully connected layer) from scratch: the math, shapes, forward pass, and backpropagation.

---

## 1. What Is a Linear Layer?

A **linear layer** applies an **affine transformation** to its input:

- **Weights** *W*: a matrix that mixes the input dimensions.
- **Bias** *b*: a vector added after the matrix multiply (one value per output dimension).

So each output is a **linear combination of all inputs, plus a constant**. No activation function is applied inside the layer; activations (ReLU, sigmoid, etc.) are separate layers that you stack after linear layers.

---

## 2. Notation and Convention

We follow the convention used in your plan:

- **Row-major batches**: each **row** of a matrix is one sample.
- **Input** *x*: shape `(batch_size, in_features)`.
- **Weights** *W*: shape `(in_features, out_features)`.
- **Bias** *b*: shape `(out_features,)` or `(1, out_features)`.
- **Output** *y*: shape `(batch_size, out_features)`.

Formula:

```
y = x · W + b
```

So for one row (one sample): *y*ᵢ = *x*ᵢ*W* + *b*. The bias *b* is added to every row (broadcasting).

---

## 3. Forward Pass

**Computation:**

```
y = x · W + b
```

- *x*: `(B, I)` — B batch, I input features.
- *W*: `(I, O)` — I inputs, O outputs.
- *b*: `(O,)` — one bias per output (broadcast over the batch).
- *y*: `(B, O)`.

**What to store for the backward pass:**

- *x* (input to the layer).
- *W* (or at least you have it as a parameter; you'll need it for ∂L/∂*x*).

You do **not** need to store *y* for the linear layer backward; you only need the **gradient of the loss w.r.t. *y*** (which the next layer or loss will give you).

---

## 4. Backward Pass (Gradients)

The backward pass receives **∂L/∂*y*** (same shape as *y*: `(B, O)`). You must compute:

1. $∂L/∂W$ — to update weights.
2. $∂L/∂b $— to update bias.
3. $\partial{L}/∂x $— to pass gradient to the previous layer.

Use the **chain rule** and the fact that *y* = *x* *W* + *b*.

### 4.1 Gradient w.r.t. weights *W*

*y* = *x* *W* + *b* ⇒ each element of *y* depends on *W* only through the matrix product *x* *W*.

```
∂L/∂W = xᵀ · (∂L/∂y)
```

- *x*ᵀ: `(I, B)`
- ∂L/∂*y*: `(B, O)`
- *x*ᵀ · (∂L/∂*y*): `(I, O)` ⇒ same shape as *W*. ✓

So in code: `W_grad = x.transpose().matmul(grad_output)` (assuming you have a `transpose()` and `matmul` on your tensor).

### 4.2 Gradient w.r.t. bias *b*

*b* is added to every row of *x* *W*. So each *b*ⱼ influences every row of *y* in column *j*. The chain rule gives:

```
∂L/∂b = Σᵢ (∂L/∂y)ᵢ,ⱼ
```

i.e. **sum the gradient of the loss w.r.t. *y* over the batch dimension** (over rows). So ∂L/∂*b* has shape `(O,)` or `(1, O)`.

```
∂L/∂b = 1ᵀ · (∂L/∂y)
```

where **1** is a column vector of ones of length B. In practice you just **sum** `grad_output` along the batch (row) axis. Your `sum_along_axis(0)` (or equivalent) does exactly that.

### 4.3 Gradient w.r.t. input *x*

```
∂L/∂x = (∂L/∂y) · Wᵀ
```

- ∂L/∂*y*: `(B, O)`
- *W*ᵀ: `(O, I)`
- Result: `(B, I)` ⇒ same shape as *x*. ✓

So in code: `x_grad = grad_output.matmul(W.transpose())`. This is what you return so the layer before this one can continue backprop.

---

## 5. Summary Table

| Quantity   | Shape   | Formula |
|------------|---------|--------|
| *x*        | (B, I)  | input  |
| *W*        | (I, O)  | weights |
| *b*        | (O,)    | bias   |
| *y*        | (B, O)  | *x* *W* + *b* |
| ∂L/∂*y*    | (B, O)  | from next layer |
| ∂L/∂*W*    | (I, O)  | *x*ᵀ · (∂L/∂*y*) |
| ∂L/∂*b*    | (O,)    | sum of ∂L/∂*y* over rows |
| ∂L/∂*x*    | (B, I)  | (∂L/∂*y*) · *W*ᵀ |

---

## 6. Mapping to Your Tensor API

From your `tensor.hpp` you have:

- `matmul(other)` — matrix multiply.
- `transpose()` — in-place transpose (and you can do `x.transpose()` conceptually; if it's in-place, you may need a copy or a non-mutating version for expressions like `x.T().matmul(grad)`).
- `sum_along_axis(axis)` — sum along a given axis (use the batch axis for ∂L/∂*b*).

So:

- **Forward:** `y = x.matmul(W) + b` (with *b* broadcast over rows).
- **Backward:**
  - `dL/dW` = `x.transpose().matmul(grad_output)` (ensure shapes (I,B) @ (B,O) = (I,O)).
  - `dL/db` = `grad_output.sum_along_axis(0)` (or axis that is the batch dimension).
  - `dL/dx` = `grad_output.matmul(W.transpose())`.

If your `transpose()` is in-place and returns a reference, you'll need to be careful not to overwrite *x* or *W* when you still need them. A separate "transposed copy" helper or a non-mutating `transposed() const` can help.

---

## 7. Intuition

- **Forward:** Each output neuron is a weighted sum of all input neurons plus a bias. *W* encodes "how much each input feature contributes to each output feature."
- **Backward:**
  - ∂L/∂*W* uses *x*ᵀ and ∂L/∂*y*: "which inputs (rows) and which output gradients (cols) pair up" to update each weight.
  - ∂L/∂*b* is the sum over the batch: the bias is shared across samples, so you accumulate its effect over all samples.
  - ∂L/∂*x* pushes the output gradient back through *W*: "how much each input dimension should change" given the output gradients.

---

## 8. Optional: One Neuron, Then Full Layer

**Single output neuron (no batch):**  
*y* = *w*₁*x*₁ + *w*₂*x*₂ + … + *w*ₙ*x*ₙ + *b*. So ∂*y*/∂*w*ᵢ = *x*ᵢ, ∂*y*/∂*b* = 1. With loss *L*(*y*): ∂*L*/∂*w*ᵢ = (∂*L*/∂*y*) · *x*ᵢ, ∂*L*/∂*b* = ∂*L*/∂*y*.

**Many neurons and batch:** the matrix form *y* = *x* *W* + *b* and the formulas above are the same idea, written in matrix notation so one matmul gives all partial derivatives at once.

---

## 9. What to Implement

1. **Forward:** Given `x`, compute `y = x * W + b`; store what backward needs (e.g. `x`, and keep `W` as a parameter).
2. **Backward:** Given `grad_output` (dL/dy), compute and store or return:
   - `grad_W = x.T @ grad_output`
   - `grad_b = sum(grad_output, axis=batch)`
   - `grad_x = grad_output @ W.T` (return this to the previous layer).

After that you can plug this layer into your `NeuralNetwork`, hook it to an optimizer for *W* and *b*, and test with `test_linear_layer.cpp` and gradient checks.
