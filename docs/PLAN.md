# Neural Network From Scratch - C++ Implementation Plan

## Learning-First Requirement

**You will implement everything yourself.** The AI assistant will **guide** you—explaining concepts, suggesting next steps, answering questions, and reviewing your code—but will **not** write the implementation for you. The goal is to learn through doing.

When you ask for help:

- You will receive explanations, formulas, and hints
- You will be pointed to relevant theory or resources
- You will be asked to try implementing before receiving more detail
- Code reviews will focus on understanding and correctness, not on providing ready-made solutions

---

## 1. Theory Foundation

Create a `docs/` folder with theory documentation that you can reference while implementing:

**docs/THEORY.md** - Core concepts:

- **Neural network basics**: Perceptron, layers, forward pass (linear transform + activation)
- **Loss functions**: MSE for regression; cross-entropy for classification (paired with softmax on the final layer)
- **Backpropagation**: Chain rule, computational graph, gradient flow from loss to each weight
- **Automatic differentiation**: Forward vs reverse mode; backprop is reverse-mode AD applied to the forward pass computation graph
- **Optimizers**: SGD, Momentum, AdaGrad, RMSprop, Adam (formulas and intuition)

**docs/AUTODIFF.md** - Dedicated AD explanation:

- What AD is: computing exact derivatives by applying chain rule to the computation graph
- Why manual backprop = reverse-mode AD: you define `backward()` for each operation that propagates gradients
- Computation graph mental model: each op stores inputs for backward pass

---

## 2. Project Architecture

**Library structure**: Create a **library** target (e.g., `neural_impl_lib`) containing the core code. The main executable, tests, and Python bindings all link against it. Do not link tests or bindings to the executable.

**Recommended structure (with tests):**

```
include/
  tensor.hpp          # Tensor interface + Eigen backend
  tensor_gpu.hpp      # GPU backend (cuBLAS + CUDA kernels)
  layer.hpp           # Abstract layer interface
  linear_layer.hpp
  activation.hpp      # ReLU, Sigmoid, Softmax (for classification output)
  loss.hpp            # MSE, CrossEntropy
  neural_network.hpp  # Composes layers, forward/backward
  optimizer.hpp       # Base optimizer
  sgd.hpp
  momentum.hpp
  rmsprop.hpp
  adam.hpp
src/
  tensor.cpp
  tensor_gpu.cpp      # GPU impl (if USE_GPU)
  linear_layer.cpp
  ...
tests/
  test_tensor.cpp
  test_linear_layer.cpp
  test_activation.cpp
  test_loss.cpp
  test_neural_network.cpp
  test_optimizer.cpp
  test_gradient_check.cpp
bindings/
  bindings.cpp        # Python module (when BUILD_PYTHON_BINDINGS)
```

---

## 3. Automatic Differentiation via Manual Backprop

**Key insight**: Manual backpropagation *is* automatic differentiation. You implement it by:

1. **Forward pass**: Each layer computes output and stores inputs (and any values needed for backward) in member variables
2. **Backward pass**: Each layer receives `dL/d_output` and computes `dL/d_input` and `dL/d_weights` using the chain rule

**Example - Linear layer** `y = xW + b`:

- Forward: store `x`, `W` (or just what you need)
- Backward: `dL/dW = x^T @ dL/dy`, `dL/dx = dL/dy @ W^T`, `dL/db = sum(dL/dy)`

**Example - ReLU** `y = max(0, x)`:

- Forward: store mask `x > 0`
- Backward: `dL/dx = dL/dy * mask` (gradient flows only where x > 0)

You will implement `backward(const Tensor& grad_output)` for each layer and loss.

---

## 4. Tensor / Linear Algebra and GPU Abstraction

**CPU backend (default)**: Use **Eigen** for matrix operations.

**GPU backend (optional)**: Build with `-DUSE_GPU=ON` to enable **cuBLAS + CUDA** implementation. See Section 13.

---

## 5. Optimizer Hierarchy

| Optimizer    | Update rule                                           | Builds on          |
| ------------ | ----------------------------------------------------- | ------------------ |
| **SGD**      | `w -= lr * grad`                                      | —                  |
| **Momentum** | `v = beta*v + grad`, `w -= lr*v`                      | SGD                |
| **RMSprop**  | `s = beta*s + grad²`, `w -= lr * grad/sqrt(s+eps)`    | —                  |
| **Adam**     | Combines Momentum (m) + RMSprop (v) + bias correction | Momentum + RMSprop |

**Adam formula** (from paper):

```
m_t = β1*m_{t-1} + (1-β1)*grad
v_t = β2*v_{t-1} + (1-β2)*grad²
m̂ = m / (1-β1^t)
v̂ = v / (1-β2^t)
w -= lr * m̂ / (sqrt(v̂) + ε)
```

---

## 6. Training Loop

```cpp
for (epoch : epochs) {
    for (batch : data) {
        output = nn.forward(batch.input);
        loss = loss_fn(output, batch.target);
        grad_loss = loss_fn.backward();
        nn.backward(grad_loss);
        optimizer.step(nn.parameters(), nn.gradients());
    }
}
```

---

## 7. Testing Strategy (Catch2 + FakeIt)

- **Catch2 v3** for unit tests
- **FakeIt** (optional) for mocking when testing classes with dependencies
- Add validation for invalid inputs and dimension mismatches

### Tests per Implementation Step

| Step           | Test file                 | What to verify                                                |
| -------------- | ------------------------- | ------------------------------------------------------------- |
| Tensor         | `test_tensor.cpp`         | matmul result, transpose, shape, element-wise ops             |
| Linear layer   | `test_linear_layer.cpp`   | forward output shape, backward gradient shape; gradient check |
| ReLU, Sigmoid, Softmax | `test_activation.cpp` | ReLU: zeros stay zero, positives pass; Sigmoid: range (0,1); Softmax: outputs sum to 1 |
| MSE, CrossEntropy | `test_loss.cpp`       | MSE: regression; CrossEntropy: classification (with softmax) |
| NeuralNetwork  | `test_neural_network.cpp` | forward/backward shapes                                       |
| Optimizers     | `test_optimizer.cpp`      | SGD/Momentum/Adam update formulas with known inputs           |
| Gradient check | `test_gradient_check.cpp` | Compare backward pass to numerical gradients (finite diff)     |

---

## 8. Implementation Order

1. **Tensor/Matrix** – Eigen integration or minimal Tensor class (CPU) → `test_tensor.cpp`
2. **Linear layer** – forward + backward → `test_linear_layer.cpp`
3. **ReLU, Sigmoid, Softmax** – forward + backward for all; Softmax for classification output layer → `test_activation.cpp`
4. **MSE and CrossEntropy loss** – MSE for regression; CrossEntropy for classification (with softmax) → `test_loss.cpp`
5. **NeuralNetwork** – stack layers, forward, backward, parameter collection → `test_neural_network.cpp`
6. **SGD optimizer** – basic training loop → `test_optimizer.cpp`
7. **Momentum** – add velocity
8. **RMSprop** – add squared gradient accumulation
9. **Adam** – combine m and v with bias correction
10. **Gradient check** – `test_gradient_check.cpp` (validates backprop)
11. **Demo** – XOR or MNIST subset to validate
12. **Python bindings** – Expose NeuralNetwork, layers, optimizers to Python via pybind11 or nanobind
13. **Model save/load** – `save(path)` and `load(path)` for NeuralNetwork
14. **GPU backend** (optional) – Tensor abstraction + cuBLAS + custom CUDA kernels

---

## 9. Validation and Model Persistence

- **XOR problem**: 2-4-1 network with sigmoid, MSE or BCE. Should converge in hundreds of epochs.
- **Classification (e.g. MNIST)**: Final layer Linear → Softmax, loss CrossEntropy; outputs are class probabilities.
- **Linear regression**: 1-1 network, verify it matches analytical solution.
- **Gradient check**: Compare your backward pass to numerical gradients (finite differences) for small nets.
- **Model save/load**: Implement `save(path)` and `load(path)` for `NeuralNetwork`.

---

## 10. Python Bindings (Higher Priority)

Python bindings are implemented **before** the GPU backend. Use **pybind11** or **nanobind**. Expose: `Tensor`, `NeuralNetwork`, `Optimizer`, training helper. Add `pytest` tests in `python/tests/`.

---

## 11. GPU Implementation (cuBLAS + CUDA) – Final Optional Step

**Chosen backend**: **cuBLAS** for NVIDIA GPUs. Build with `-DUSE_GPU=ON`. cuBLAS handles GEMM; custom CUDA kernels handle ReLU, element-wise ops, sum.
