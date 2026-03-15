# Backpropagation Walkthrough — A Small Network Example

This document walks through **backpropagation** step by step using a tiny neural network with concrete numbers. The goal is to see how gradients flow backward through the layers and how the chain rule connects everything.

---

## 1. The Network

We use a minimal network:

```
  x (input)  →  Linear₁  →  h  →  Linear₂  →  ŷ (output)  →  Loss L
```

**Dimensions:**

| Variable | Shape | Description |
|----------|-------|-------------|
| x | (1, 2) | 1 sample, 2 input features |
| W₁ | (2, 2) | Linear₁ weights |
| b₁ | (2, 1) | Linear₁ bias |
| h | (1, 2) | Hidden layer output |
| W₂ | (2, 1) | Linear₂ weights |
| b₂ | (1, 1) | Linear₂ bias |
| ŷ | (1, 1) | Final output (prediction) |
| y | (1, 1) | Target (ground truth) |
| L | scalar | Loss |

**Loss:** Mean Squared Error (MSE) for one sample: $L = \frac{1}{2}(\hat{y} - y)^2$

---

## 2. Forward Pass (Concrete Numbers)

Let’s fix concrete values and trace the forward pass.

### 2.1 Setup

```
x = [1, 2]           shape (1, 2)

W₁ = [ 0.5  0.3 ]   shape (2, 2)
     [ 0.2  0.4 ]

b₁ = [ 0.1 ]         shape (2, 1)
     [ 0.2 ]

W₂ = [ 0.6 ]         shape (2, 1)
     [ 0.7 ]

b₂ = [ 0.1 ]         shape (1, 1)

y = [ 2 ]            target (1, 1)
```

### 2.2 Layer 1: h = x·W₁ + b₁

```
h = x · W₁ + b₁
  = [1, 2] · [ 0.5  0.3 ] + [ 0.1 ]
             [ 0.2  0.4 ]   [ 0.2 ]

  = [ 1·0.5 + 2·0.2,  1·0.3 + 2·0.4 ] + [ 0.1, 0.2 ]
  = [ 0.9, 1.1 ] + [ 0.1, 0.2 ]
  = [ 1.0, 1.3 ]
```

So **h = [1.0, 1.3]** with shape (1, 2).

### 2.3 Layer 2: ŷ = h·W₂ + b₂

```
ŷ = h · W₂ + b₂
  = [1.0, 1.3] · [ 0.6 ] + [ 0.1 ]
                 [ 0.7 ]

  = [ 1.0·0.6 + 1.3·0.7 ] + [ 0.1 ]
  = [ 0.6 + 0.91 ] + 0.1
  = [ 1.61 ]
```

So **ŷ = [1.61]** with shape (1, 1).

### 2.4 Loss: L = ½(ŷ − y)²

```
L = ½(1.61 − 2)² = ½(0.39)² = ½ · 0.1521 ≈ 0.076
```

---

## 3. The Chain Rule (What We Need)

Backpropagation uses the **chain rule** to compute gradients. For each quantity we want a gradient for, we express it in terms of the gradient we already have.

**Flow of gradients (backward):**

```
∂L/∂ŷ  →  ∂L/∂h  →  ∂L/∂x
         ↘ ∂L/∂W₂, ∂L/∂b₂
         ↘ ∂L/∂W₁, ∂L/∂b₁
```

We start with **∂L/∂ŷ** (gradient of loss w.r.t. output) and propagate backward.

---

## 4. Backward Pass — Step by Step

### 4.1 Gradient of the Loss: ∂L/∂ŷ

$L = \frac{1}{2}(\hat{y} - y)^2$

$$
\frac{\partial L}{\partial \hat{y}} = \hat{y} - y = 1.61 - 2 = -0.39
$$

So **∂L/∂ŷ = [-0.39]** with shape (1, 1). This is the `grad_output` for Linear₂.

---

### 4.2 Linear₂ Backward

Linear₂ computes: **ŷ = h·W₂ + b₂**

We need:
- **∂L/∂h** — to pass to Linear₁
- **∂L/∂W₂** — to update W₂
- **∂L/∂b₂** — to update b₂

**From the theory:**

| Gradient | Formula | Shapes |
|----------|---------|--------|
| ∂L/∂h | (∂L/∂ŷ) · W₂ᵀ | (1,1)×(1,2) = (1,2) |
| ∂L/∂W₂ | hᵀ · (∂L/∂ŷ) | (2,1)×(1,1) = (2,1) |
| ∂L/∂b₂ | sum(∂L/∂ŷ) over batch | (1,1) |

**Compute:**

```
∂L/∂h = (∂L/∂ŷ) · W₂ᵀ
      = [-0.39] · [ 0.6  0.7 ]
      = [ -0.39·0.6,  -0.39·0.7 ]
      = [ -0.234,  -0.273 ]
```

```
∂L/∂W₂ = hᵀ · (∂L/∂ŷ)
       = [ 1.0 ] · [-0.39]
         [ 1.3 ]
       = [ -0.39 ]
         [ -0.507 ]
```

```
∂L/∂b₂ = sum over batch of ∂L/∂ŷ = -0.39
```

**Summary for Linear₂:**
- `grad_input` (to pass back): **∂L/∂h = [-0.234, -0.273]**
- `grad_weights`: **∂L/∂W₂ = [-0.39, -0.507]ᵀ**
- `grad_bias`: **∂L/∂b₂ = [-0.39]**

---

### 4.3 Linear₁ Backward

Linear₁ computes: **h = x·W₁ + b₁**

We receive **∂L/∂h = [-0.234, -0.273]** from Linear₂. This is our `grad_output`.

We need:
- **∂L/∂x** — to pass to the input (or previous layer)
- **∂L/∂W₁** — to update W₁
- **∂L/∂b₁** — to update b₁

**Formulas:**

| Gradient | Formula | Shapes |
|----------|---------|--------|
| ∂L/∂x | (∂L/∂h) · W₁ᵀ | (1,2)×(2,2) = (1,2) |
| ∂L/∂W₁ | xᵀ · (∂L/∂h) | (2,1)×(1,2) = (2,2) |
| ∂L/∂b₁ | sum(∂L/∂h) over batch | (2,1) |

**Compute:**

```
∂L/∂x = (∂L/∂h) · W₁ᵀ
      = [ -0.234,  -0.273 ] · [ 0.5  0.2 ]
                              [ 0.3  0.4 ]
      = [ -0.234·0.5 - 0.273·0.3,  -0.234·0.2 - 0.273·0.4 ]
      = [ -0.117 - 0.082,  -0.047 - 0.109 ]
      = [ -0.199,  -0.156 ]
```

```
∂L/∂W₁ = xᵀ · (∂L/∂h)
       = [ 1 ] · [ -0.234  -0.273 ]
         [ 2 ]
       = [ -0.234   -0.273 ]
         [ -0.468   -0.546 ]
```

```
∂L/∂b₁ = sum over batch of ∂L/∂h

With batch size 1, ∂L/∂h is (1,2). Sum over batch gives (1,2).
To match b₁ shape (2,1), we transpose: ∂L/∂b₁ = [ -0.234 ]
                                                 [ -0.273 ]
```

---

## 5. Summary: The Full Backward Flow

Gradients flow from the loss back to the input. Each layer receives a gradient, computes three things, and passes one of them backward.

```
  Loss L
     │
     │  ∂L/∂ŷ = ŷ − y  (from loss derivative)
     ▼
  Linear₂
     │  Receives ∂L/∂ŷ.  Computes:
     │    • ∂L/∂h  = (∂L/∂ŷ)·W₂ᵀ     → passed to Linear₁
     │    • ∂L/∂W₂ = hᵀ·(∂L/∂ŷ)      → for optimizer
     │    • ∂L/∂b₂ = sum(∂L/∂ŷ)      → for optimizer
     ▼
  ∂L/∂h
     │
  Linear₁
     │  Receives ∂L/∂h.  Computes:
     │    • ∂L/∂x  = (∂L/∂h)·W₁ᵀ     → passed to input
     │    • ∂L/∂W₁ = xᵀ·(∂L/∂h)      → for optimizer
     │    • ∂L/∂b₁ = sum(∂L/∂h)      → for optimizer
     ▼
  Input x
```

**Key idea:** Each layer receives **∂L/∂(its output)** and computes:
1. **∂L/∂(its input)** — passed backward to the previous layer
2. **∂L/∂W** and **∂L/∂b** — stored for the optimizer to update parameters

---

## 6. Mapping to Your Code

| Formula | Your `LinearLayer::backward` |
|---------|------------------------------|
| ∂L/∂W = xᵀ · (∂L/∂y) | `m_grad_weights = m_input.transpose().matmul(grad_output)` |
| ∂L/∂b = sum(∂L/∂y) over batch | `m_grad_bias = grad_output.sum_along_axis(0).transpose()` |
| ∂L/∂x = (∂L/∂y) · Wᵀ | `return grad_output.matmul(m_weights.transpose())` |

**Why we cache `m_input`:** We need **x** (the input) to compute ∂L/∂W. The forward pass stores it; backward uses it.

**Why backward doesn’t update weights:** The layer only computes gradients. The optimizer (e.g. SGD) does `W -= lr * grad_W` and `b -= lr * grad_b`.

---

## 7. The Chain Rule in One Picture

For a composition $L = f(g(h(x)))$:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}
$$

Backpropagation computes this product **from right to left** (output → input). Each layer multiplies the incoming gradient by its local Jacobian. The matrix forms (e.g. ∂L/∂x = (∂L/∂y)·Wᵀ) are exactly these chain-rule products.
