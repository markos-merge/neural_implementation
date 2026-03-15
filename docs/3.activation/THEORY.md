# Activation Layers — Theory

This document covers what you need to know to implement **activation functions** from scratch: the math, shapes, forward pass, and backpropagation for ReLU and Sigmoid.

---

## 1. What Are Activation Layers?

Activation layers apply **element-wise, non-linear** transformations. They sit after linear layers. A stack of linear layers alone is equivalent to one linear layer; activations break that and let the network learn non-linear functions.

---

## 2. ReLU (Rectified Linear Unit)

### 2.1 Definition

ReLU clips negative values to zero:

```
y = max(0, x)  =  x if x > 0, else 0
```

Element-wise: each output equals the input if positive, else zero.

### 2.2 Notation and Shapes

- **Input** *x*: shape `(batch_size, features)`.
- **Output** *y*: shape `(batch_size, features)` — unchanged.

### 2.3 Forward Pass

**Computation:**

```
yᵢ,ⱼ = max(0, xᵢ,ⱼ)
```

**What to store for backward:**

- A **mask**: 1 where *x* > 0, 0 elsewhere. Same shape as *x*.

### 2.4 Backward Pass

The derivative of ReLU w.r.t. *x*:

```
∂y/∂x = 1 if x > 0, else 0
```

By the chain rule, given ∂L/∂*y* from the next layer:

```
∂L/∂x = (∂L/∂y) · (∂y/∂x)
```

Since ∂y/∂x is the mask, this becomes **element-wise multiplication** of ∂L/∂*y* with the mask. Gradient flows only where *x* > 0.

### 2.5 Summary Table (ReLU)

| Quantity      | Shape   | Formula                          |
|---------------|---------|----------------------------------|
| *x*           | (B, F)  | input                            |
| *y*           | (B, F)  | max(0, *x*)                      |
| mask          | (B, F)  | 1 where *x* > 0, 0 elsewhere     |
| ∂L/∂*y*       | (B, F)  | from next layer                  |
| ∂L/∂*x*       | (B, F)  | (∂L/∂*y*) ⊙ mask                 |

---

## 3. Sigmoid

### 3.1 Definition

```
σ(x) = 1 / (1 + e⁻ˣ)
```

Output range: (0, 1). Often used for probabilities.

### 3.2 Notation and Shapes

- **Input** *x*: shape `(batch_size, features)`.
- **Output** *y*: shape `(batch_size, features)` — unchanged.

### 3.3 Forward Pass

**Computation:**

```
yᵢ,ⱼ = 1 / (1 + exp(-xᵢ,ⱼ))
```

**What to store for backward:**

- *y* (the output). The derivative can be written in terms of *y* alone.

### 3.4 Backward Pass

A useful identity:

```
dσ/dx = σ(x) · (1 - σ(x)) = y · (1 - y)
```

By the chain rule:

```
∂L/∂x = (∂L/∂y) · y · (1 - y)
```

Element-wise multiplication.

### 3.5 Summary Table (Sigmoid)

| Quantity      | Shape   | Formula                          |
|---------------|---------|----------------------------------|
| *x*           | (B, F)  | input                            |
| *y*           | (B, F)  | 1 / (1 + e⁻ˣ)                    |
| ∂L/∂*y*       | (B, F)  | from next layer                  |
| ∂L/∂*x*       | (B, F)  | (∂L/∂*y*) ⊙ y ⊙ (1 - y)         |

---

## 4. Mapping to Your Tensor API

You need element-wise operations:

- **ReLU forward:** *y* = max(0, *x*) — element-wise max with 0.
- **ReLU backward:** ∂L/∂*x* = (∂L/∂*y*) ⊙ mask — element-wise multiply.
- **Sigmoid forward:** *y* = 1/(1 + exp(-*x*)) — element-wise sigmoid.
- **Sigmoid backward:** ∂L/∂*x* = (∂L/∂*y*) ⊙ *y* ⊙ (1 - *y*) — element-wise multiply.

Check your `tensor.hpp` for element-wise product (`operator*`), max, comparison, and exp.

---

## 5. Intuition

- **ReLU:** Keeps positives, zeros negatives. Simple and fast. Neurons that stay ≤ 0 get zero gradient ("dead" ReLUs).
- **Sigmoid:** Squashes to (0, 1). Saturates for large |*x*|; gradients can vanish.

---

## 6. What to Implement

1. **ReLU:** Forward: *y* = max(0, *x*); store mask (*x* > 0). Backward: return (∂L/∂*y*) ⊙ mask.
2. **Sigmoid:** Forward: *y* = σ(*x*); store *y*. Backward: return (∂L/∂*y*) ⊙ *y* ⊙ (1 - *y*).
