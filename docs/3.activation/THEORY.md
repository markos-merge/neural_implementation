# Activation Layers — Theory

This document covers what you need to know to implement **activation functions** from scratch: the math, shapes, forward pass, and backpropagation for ReLU, Sigmoid, and Softmax. It also defines **cross-entropy** and explains why **softmax + cross-entropy** is usually implemented as a single, numerically stable path (§4.6–§4.7).

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

## 4. Softmax

Softmax turns a **vector of logits** (raw scores) into a **probability distribution**: all outputs are positive and sum to 1. Unlike ReLU and Sigmoid, it is **not** element-wise in isolation—it couples all components of a vector.

**Notation (same as §2–3):** **x** is the layer **input** and **y** the **output**. For softmax, each $x_i$ is a logit (class score); $y_i$ is the corresponding probability after normalization.

### 4.1 Definition (one sample)

For one row of class scores $\mathbf{x} \in \mathbb{R}^C$:

$$
y_i = \frac{e^{x_i}}{\sum_{k=1}^{C} e^{x_k}}, \quad i = 1,\ldots,C
$$

Properties:

- $y_i > 0$ for all $i$ (strictly positive if all $x_i$ are finite).
- $\sum_{i=1}^{C} y_i = 1$.
- **Translation invariance:** adding the same constant $c$ to every $x_i$ does not change $\mathbf{y}$ (the exponentials pick up a common factor that cancels in the ratio).

### 4.2 Batch layout

In code, tensors are usually **(batch, classes)**:

- **Input** *x*: shape `(batch_size, num_classes)` — each **row** is one sample’s logits.
- **Output** *y*: same shape — apply the softmax formula **independently to each row** (softmax along the class dimension).

So for batch index $b$:

$$
y_{b,i} = \frac{e^{x_{b,i}}}{\sum_{k=1}^{C} e^{x_{b,k}}}
$$

### 4.3 Numerical stability

Directly computing $e^{x_i}$ can overflow for large logits. The usual trick uses translation invariance: subtract the row maximum before exponentiating.

Define $m_b = \max_j x_{b,j}$. Then:

$$
y_{b,i} = \frac{e^{x_{b,i} - m_b}}{\sum_{k} e^{x_{b,k} - m_b}}
$$

All exponentials are $\leq 1$ in that row, which avoids huge magnitudes while giving the same *y* as the naive formula.

### 4.4 Jacobian of softmax

For one row, let $\mathbf{y} = \mathrm{softmax}(\mathbf{x})$. Partial derivatives:

$$
\frac{\partial y_i}{\partial x_j} = y_i \left( \delta_{ij} - y_j \right)
$$

where $\delta_{ij}$ is 1 if $i=j$ and 0 otherwise. So the Jacobian $\mathbf{J}$ with entries $J_{ij} = \partial y_i / \partial x_j$ is **not** diagonal: changing one input moves several outputs.

### 4.5 Backward pass (generic upstream gradient)

Given a scalar loss $L$ and upstream gradient $\frac{\partial L}{\partial \mathbf{y}}$ (same shape as $\mathbf{y}$, row-wise), the gradient w.r.t. the input is:

$$
\frac{\partial L}{\partial x_j} = \sum_i \frac{\partial L}{\partial y_i} \frac{\partial y_i}{\partial x_j}
= y_j \left( \frac{\partial L}{\partial y_j} - \sum_i \frac{\partial L}{\partial y_i} \, y_i \right)
$$

Vector form for one row (using $\odot$ for element-wise product):

$$
\frac{\partial L}{\partial \mathbf{x}} = \mathbf{y} \odot \left( \frac{\partial L}{\partial \mathbf{y}} - \left( \frac{\partial L}{\partial \mathbf{y}} \cdot \mathbf{y} \right) \mathbf{1} \right)
$$

Here $\frac{\partial L}{\partial \mathbf{y}} \cdot \mathbf{y}$ is a scalar (dot product); subtract it from each component of $\frac{\partial L}{\partial \mathbf{y}}$, then multiply element-wise by $\mathbf{y}$.

**What to store for backward:** the forward output *y* (softmax probabilities).

### 4.6 Cross-entropy (definition)

Cross-entropy compares two discrete distributions over the same outcomes. Let $q$ be the **target** distribution and $p$ the **model** distribution over classes $i = 1,\ldots,C$, with $q_i \ge 0$, $p_i \ge 0$, and $\sum_i q_i = \sum_i p_i = 1$.

The **cross-entropy** of $p$ relative to $q$ is:

$$
H(q, p) = -\sum_{i=1}^{C} q_i \log p_i
$$

(Use natural log for “nats”; $\log_2$ gives bits.) Intuitively: if labels are drawn from $q$ and you assign probability $p_i$ to class $i$, then $H(q,p)$ is the expected negative log-likelihood of your model under those draws.

**Relation to entropy and KL divergence:**

$$
H(q, p) = H(q) + D_{\mathrm{KL}}(q \,\|\, p), \quad H(q) = -\sum_i q_i \log q_i
$$

For fixed $q$, minimizing $H(q,p)$ in $p$ is equivalent to minimizing $D_{\mathrm{KL}}(q \| p)$.

**One-hot targets (multiclass classification).** If the true label is always a single class $c$ (one-hot $q$ with $q_c = 1$ and $q_i = 0$ for $i \ne c$), the sum collapses to:

$$
H(q, p) = -\log p_c
$$

That is the usual **categorical cross-entropy loss** for one sample: negative log probability of the correct class.

In a classifier, $p$ is often $\mathrm{softmax}(\mathbf{x})$ applied to logits $\mathbf{x}$. The next subsection explains why **softmax and this loss are often implemented together**.

### 4.7 Softmax + cross-entropy (why it is a special case)

Let $\mathbf{t}$ be a one-hot label vector for the true class, and $\mathbf{y} = \mathrm{softmax}(\mathbf{x})$ the predicted probabilities. The loss is the cross-entropy from §4.6:

$$
L = -\sum_i t_i \log y_i
$$

**Gradient w.r.t. logits.** With this pairing, the gradient w.r.t. the input logits simplifies to:

$$
\frac{\partial L}{\partial \mathbf{x}} = \mathbf{y} - \mathbf{t}
$$

So the backward pass through “softmax + cross-entropy” is often implemented as **prediction minus label**, row-wise. That is **not** a general rule for softmax: it holds because the upstream loss is exactly cross-entropy with a one-hot target. For a different loss on $\mathbf{y}$, you still use the generic softmax Jacobian in §4.5.

**Numerical stability.** Computing $\mathbf{y}$ and then $\log \mathbf{y}$ can lose precision (very small $y_i$, $\log$ of tiny values). In practice, the loss is evaluated in **log space** using the log-sum-exp form:

$$
\log y_i = x_i - \log\sum_k e^{x_k}
$$

Subtracting $\max_k x_k$ inside the sum (same trick as §4.3) keeps exponentials bounded. Frameworks often expose this as **log-softmax** plus **negative log-likelihood**, or a single **cross-entropy from logits** op that never materializes probabilities unnecessarily.

**Summary:** Cross-entropy is defined between distributions (§4.6). Pairing it with softmax on logits gives a stable forward pass and a simple backward ($\mathbf{y} - \mathbf{t}$); that is why softmax+CE is treated as one fused case rather than two independent layers.

### 4.8 Summary table (Softmax, per row)

| Quantity | Shape | Role |
|----------|--------|------|
| *x* | (1, C) or row of (B, C) | input (logits) |
| *y* | same | $\mathrm{softmax}(\mathbf{x})$, row sums to 1 |
| $\partial L/\partial \mathbf{y}$ | same | upstream gradient |
| $\partial L/\partial \mathbf{x}$ | same | $\mathbf{y} \odot (\partial L/\partial \mathbf{y} - (\partial L/\partial \mathbf{y}\cdot\mathbf{y})\mathbf{1})$ |

---

## 5. Mapping to Your Tensor API

You need element-wise operations:

- **ReLU forward:** *y* = max(0, *x*) — element-wise max with 0.
- **ReLU backward:** ∂L/∂*x* = (∂L/∂*y*) ⊙ mask — element-wise multiply.
- **Sigmoid forward:** *y* = 1/(1 + exp(-*x*)) — element-wise sigmoid.
- **Sigmoid backward:** ∂L/∂*x* = (∂L/∂*y*) ⊙ *y* ⊙ (1 - *y*) — element-wise multiply.

Check your `tensor.hpp` for element-wise product (`operator*`), max, comparison, and exp. Softmax needs **row-wise** normalization: exponentiate, sum across the class axis, divide each row by its sum (or equivalent stable formulation).

---

## 6. Intuition

- **ReLU:** Keeps positives, zeros negatives. Simple and fast. Neurons that stay ≤ 0 get zero gradient ("dead" ReLUs).
- **Sigmoid:** Squashes to (0, 1). Saturates for large |*x*|; gradients can vanish.
- **Softmax:** Turns a vector of scores into a probability vector (sums to 1). Couples all classes in one row; backward needs the full Jacobian, or combine with cross-entropy for the simpler $\mathbf{y} - \mathbf{t}$ gradient (§4.7).

---

## 7. What to Implement

1. **ReLU:** Forward: *y* = max(0, *x*); store mask (*x* > 0). Backward: return (∂L/∂*y*) ⊙ mask.
2. **Sigmoid:** Forward: *y* = σ(*x*); store *y*. Backward: return (∂L/∂*y*) ⊙ *y* ⊙ (1 - *y*).
3. **Softmax:** Forward: *y* = softmax(*x*) per row; store *y*. Backward: use the Jacobian formula in §4.5 (or fused softmax + cross-entropy in §4.7). Consider subtracting row max before exp for stability (§4.3).
