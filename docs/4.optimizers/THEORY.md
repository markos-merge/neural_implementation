# Optimizers — Theory

This document covers what you need to know to implement **optimizers** from scratch: the math, update rules, and intuition for SGD, Momentum, RMSprop, and Adam.

---

## 1. What Are Optimizers?

After backpropagation, each trainable parameter has a **gradient** $\nabla L$ (the direction of steepest ascent of the loss). Optimizers decide **how** to update parameters given these gradients. The goal: minimize the loss by moving parameters in the direction of steepest *descent* (opposite to the gradient).

---

## 2. Vanilla SGD (Stochastic Gradient Descent)

### 2.1 Update Rule

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla L(\theta_t)
$$

- $\theta$: parameter (e.g. weights, bias)
- $\eta$: learning rate (hyperparameter)
- $\nabla L$: gradient of the loss w.r.t. $\theta$

### 2.2 Intuition

Take a step in the direction that *reduces* the loss. The learning rate controls step size: too large → overshoot; too small → slow convergence.

### 2.3 Summary

| Symbol | Meaning |
|--------|---------|
| $\theta$ | Parameter to update |
| $\eta$ | Learning rate |
| $\nabla L$ | Gradient of loss w.r.t. $\theta$ |

**In code:** `param -= learning_rate * grad`

---

## 3. SGD with Momentum

### 3.1 Motivation

Vanilla SGD can oscillate in narrow valleys or progress slowly along shallow directions. **Momentum** accumulates past gradients to smooth updates and accelerate in consistent directions.

### 3.2 Update Rule

$$
v_t = \beta \cdot v_{t-1} + \nabla L(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - \eta \cdot v_t
$$

- $v$: velocity (accumulated gradient)
- $\beta$: momentum coefficient, typically 0.9 or 0.99

### 3.3 Intuition

- $v$ is an exponential moving average of gradients.
- Consistent gradients → $v$ grows → larger steps.
- Oscillating gradients → $v$ partially cancels → smaller effective step.

### 3.4 Summary Table

| Quantity | Formula |
|----------|---------|
| Velocity update | $v_t = \beta v_{t-1} + \nabla L$ |
| Parameter update | $\theta_{t+1} = \theta_t - \eta v_t$ |

**What to store:** velocity $v$ for each parameter (same shape as the parameter). Initialize $v = 0$ before training.

---

## 4. RMSprop

### 4.1 Motivation

SGD uses the same learning rate for all parameters. Some dimensions may have large gradients, others small. **RMSprop** adapts the step size per parameter using a running average of squared gradients.

### 4.2 Update Rule

$$
s_t = \beta \cdot s_{t-1} + (1 - \beta) \cdot (\nabla L)^2
$$

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\nabla L}{\sqrt{s_t} + \varepsilon}
$$

- $s$: running average of squared gradients (element-wise)
- $\beta$: decay rate, typically 0.9 or 0.99
- $\varepsilon$: small constant (e.g. $10^{-8}$) to avoid division by zero

### 4.3 Intuition

- $s$ tracks the "magnitude" of recent gradients.
- Large gradients → large $s$ → smaller effective step (divided by $\sqrt{s}$).
- Small gradients → small $s$ → larger effective step.
- Per-parameter adaptive learning rate.

### 4.4 Summary Table

| Quantity | Formula |
|----------|---------|
| Squared gradient accumulation | $s_t = \beta s_{t-1} + (1-\beta)(\nabla L)^2$ |
| Parameter update | $\theta_{t+1} = \theta_t - \eta \frac{\nabla L}{\sqrt{s_t} + \varepsilon}$ |

**What to store:** $s$ for each parameter (same shape). Initialize $s = 0$ before training.

---

## 5. Adam (Adaptive Moment Estimation)

### 5.1 Motivation

**Adam** combines **Momentum** (first moment) and **RMSprop** (second moment), plus **bias correction** for early training steps when $m$ and $v$ are initialized to zero.

### 5.2 Update Rule

$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot \nabla L
$$

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot (\nabla L)^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
$$

- $m$: first moment (momentum-like)
- $v$: second moment (RMSprop-like)
- $\beta_1$: typically 0.9
- $\beta_2$: typically 0.999
- $\varepsilon$: typically $10^{-8}$

### 5.3 Bias Correction

At $t = 1$, $m_1 = (1-\beta_1)\nabla L$ and $v_1 = (1-\beta_2)(\nabla L)^2$ are biased toward zero. The correction $\hat{m} = m / (1 - \beta^t)$ scales them so the expected value is correct.

### 5.4 Summary Table

| Quantity | Formula |
|----------|---------|
| First moment | $m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L$ |
| Second moment | $v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L)^2$ |
| Bias-corrected $\hat{m}$ | $\hat{m}_t = m_t / (1 - \beta_1^t)$ |
| Bias-corrected $\hat{v}$ | $\hat{v}_t = v_t / (1 - \beta_2^t)$ |
| Parameter update | $\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$ |

**What to store:** $m$, $v$, and timestep $t$ for each parameter. Initialize $m = 0$, $v = 0$, $t = 0$ before training.

---

## 6. Optimizer Hierarchy (from PLAN)

| Optimizer | Update rule | Builds on |
|-----------|-------------|-----------|
| **SGD** | $\theta \leftarrow \theta - \eta \nabla L$ | — |
| **Momentum** | $v \leftarrow \beta v + \nabla L$, $\theta \leftarrow \theta - \eta v$ | SGD |
| **RMSprop** | $s \leftarrow \beta s + (\nabla L)^2$, $\theta \leftarrow \theta - \eta \frac{\nabla L}{\sqrt{s}+\varepsilon}$ | — |
| **Adam** | Combines $m$ (momentum) + $v$ (RMSprop) + bias correction | Momentum + RMSprop |

---

## 7. Interface Design

An optimizer needs to update parameters given gradients. A minimal interface:

```
step(param, grad) → void
```

- `param`: mutable reference to the parameter tensor
- `grad`: gradient of the loss w.r.t. that parameter (computed by backprop)

Each trainable layer (e.g. `LinearLayer`) has parameters (weights, bias) and their gradients. After `backward()`, the layer calls `optimizer.step(weights, grad_weights)` and `optimizer.step(bias, grad_bias)`.

---

## 8. Implementation Order

1. **SGD** — simplest; good baseline.
2. **Momentum** — add velocity state; often improves convergence.
3. **RMSprop** — per-parameter adaptive learning rate.
4. **Adam** — combines both; default choice for many problems.

---

## 9. References

- **Sutskever et al. (2013)**: On the importance of initialization and momentum in deep learning.
- **Tieleman & Hinton (2012)**: RMSprop (course slides).
- **Kingma & Ba (2015)**: Adam: A Method for Stochastic Optimization.
