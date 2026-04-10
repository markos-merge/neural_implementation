# Neural Network From Scratch - C++ Implementation Plan

## Learning-First Requirement

**You will implement everything yourself.** The AI assistant will **guide** you—explaining concepts, suggesting next steps, answering questions, and reviewing your code—but will **not** write the implementation for you. The goal is to learn through doing.

When you ask for help:

- You will receive explanations, formulas, and hints
- You will be pointed to relevant theory or resources
- You will be asked to try implementing before receiving more detail
- Code reviews will focus on understanding and correctness, not on providing ready-made solutions

---

## Current Status

**Done (baseline):**

- Tensor / layer stack, sequential network, manual backprop for MLP blocks
- Losses (e.g. MSE, cross-entropy + softmax), **SGD** optimizer
- **GPU backend** (cuBLAS + CUDA kernels) behind your tensor/runtime abstraction
- **Validation**: MNIST and CIFAR-10 demos with fully connected models

**Current focus:** **Convolutional Neural Networks (CNNs)** — new layers, 4D tensors (NCHW), conv backward, pooling, then GPU acceleration of conv/pool (see Section 12).

**Planned for stronger CIFAR-10 results:** **Batch normalization** (conv blocks), **train-time augmentation** (random crop 32×32 + horizontal flip), **LR schedule** (cosine or step decay) with **longer training** — see §12 *Strong CIFAR-10 training recipe (planned)*.

---

## 1. Theory Foundation

Create a `docs/` folder with theory documentation that you can reference while implementing:

**docs/THEORY.md** - Core concepts:

- **Neural network basics**: Perceptron, layers, forward pass (linear transform + activation)
- **Loss functions**: MSE for regression; cross-entropy for classification (paired with softmax on the final layer)
- **Backpropagation**: Chain rule, computational graph, gradient flow from loss to each weight
- **Automatic differentiation**: Forward vs reverse mode; backprop is reverse-mode AD applied to the forward pass computation graph
- **Optimizers**: SGD, Momentum, AdaGrad, RMSprop, Adam (formulas and intuition)
- **CNNs (add for this phase)**:
  - **Convolution**: local connectivity, weight sharing, output shape vs kernel, stride, padding
  - **Pooling**: max-pooling (argmax mask for backward); optional average pooling
  - **Flatten / view**: reshaping feature maps into a vector before the final linear layer
  - **Conv backward**: gradient w.r.t. input and filters (link to im2col + GEMM or direct loops)
- **Regularization** (reduce overfitting, often smaller effective “complexity”):
  - **L2 weight decay**: add \lambda W^2 to the loss; gradient adds 2\lambda W (or absorb into optimizer as `weight_decay`)
  - **L1**: sparsity-inducing penalty \lambda W_1; subgradient / soft-threshold variants
  - **Dropout**: random zeroing of activations in training; scale at test time (inverted dropout) or average masks—needs `forward`/`backward` changes and a train/eval flag
- **Early stopping**: monitor **validation** loss or accuracy; stop when no improvement for `patience` epochs; restores best weights (needs a held-out val split and training-loop support)
- **Smaller models**: fewer / narrower layers, fewer conv channels—primary lever for **size**; advanced: **pruning** (remove small weights), **quantization** (lower precision)—document later when core training is solid

**docs/AUTODIFF.md** - Dedicated AD explanation:

- What AD is: computing exact derivatives by applying chain rule to the computation graph
- Why manual backprop = reverse-mode AD: you define `backward()` for each operation that propagates gradients
- Computation graph mental model: each op stores inputs for backward pass

---

## 2. Project Architecture

**Library structure**: Create a **library** target (e.g., `neural_impl_lib`) containing the core code. The main executable, tests, and Python bindings all link against it. Do not link tests or bindings to the executable.

**Recommended structure (with tests)** — extend your existing layout with CNN pieces:

```
include/
  tensors/            # Tensor interface; CPU and optional GPU
  layers/
    linear_layer.hpp
    relu_layer.hpp
    ...
    conv2d_layer.hpp   # NEW: 2D convolution
    max_pool2d_layer.hpp  # NEW: max pooling (store indices for backward)
    flatten_layer.hpp  # NEW: NCHW → (N, C*H*W) or equivalent
    dropout_layer.hpp  # NEW: Bernoulli mask (train); identity / scaled (eval); 2D + 4D
  nn/
    sequential_nn.hpp      # variadic template (C++ demos; optional to keep)
    sequential_module.hpp    # runtime std::vector<std::shared_ptr<Layer>> (bindings + flexible C++)
  losses/
  optimizer/
  training/           # optional: early_stopping.hpp, train_config.hpp (patience, val split)
  cuda/               # GPU kernels (existing + conv/pool later)
src/
  ...
tests/
  test_conv2d_layer.cpp      # NEW
  test_max_pool2d_layer.cpp  # NEW
  test_dropout_layer.cpp     # NEW
  test_tensor.cpp
  ...
bindings/
  bindings.cpp        # Python module (when BUILD_PYTHON_BINDINGS)
```

### Compile-time (variadic) vs runtime (polymorphic) stacks

A **variadic** sequential model (`SequentialNN<Tensor, Loss, Layers...>` with a `std::tuple` of concrete layer types) is a good fit for **C++ demos and tests**: zero-overhead wiring, shape inference at compile time, and no virtual dispatch.

It is a **poor fit for Python bindings**: each distinct layer sequence is a **different C++ type**, so pybind11 would need explicit instantiations for every architecture you want to expose, and you cannot build “`Linear` then `ReLU` then …” **at runtime** from Python without a separate API.

**Plan for bindings:** introduce a **runtime layer graph** that bindings target:

- A small **abstract `Layer` interface** (or `Module`) with `forward` / `backward` (and tensor wiring compatible with your existing `LayerBase` ideas), using `**std::shared_ptr<Layer>`** (or `unique_ptr` + custom holder) so pybind11 can own layers uniformly.
- A `**SequentialModule**` (or `Network`) holding `std::vector<std::shared_ptr<Layer>>`, wiring slots the same way you do with latches today, but dispatching through **virtual calls**.
- **Concrete layers** (`LinearLayer`, `Conv2d`, …) **inherit** the interface; C++ code can still use templates internally where useful, but the **public, bindable surface** is the polymorphic container.

You can **keep** the variadic `SequentialNN` for fast, fixed C++ experiments and **add** the runtime stack when you approach bindings—no need to delete the template version on day one.

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

**Example - MaxPool2d**:

- Forward: store **which index** won the max in each pool window (or a mask)
- Backward: route `dL/d_output` only to those positions; zeros elsewhere

**Example - Conv2d**:

- Forward: compute output feature maps; store anything needed for backward (e.g. padded input, or im2col matrices)
- Backward: `dL/d_filters`, `dL/d_input` (derivations follow from conv as a linear op over patches—often implemented via **im2col** / **col2im** plus GEMM on CPU/GPU)

**Example - Dropout** (prefer **inverted dropout**):

- Forward (train): draw mask m \sim \mathrm{Bernoulli}(p), output y = (x \odot m) / p so **eval** can be **identity** (no scaling)
- Forward (eval): y = x
- Backward (train): `dL/dx = (dL/dy ⊙ m) / p`; eval: `dL/dx = dL/dy`
- Store the **mask** from forward for backward; respect a global `**training`** flag

You will implement `backward(const Tensor& grad_output)` for each layer and loss.

---

## 4. Tensor / Linear Algebra and GPU Abstraction

**CPU backend (default)**: Use **Eigen** for matrix operations.

**GPU backend (optional)**: Build with `-DUSE_GPU=ON` to enable **cuBLAS + CUDA** implementation. Base linear algebra and elementwise ops: see Section 11. **CNN on GPU**: reuse GEMM via im2col for conv when you add it; pooling as dedicated CUDA kernels or a first CPU-atop-GPU path depending on your milestone.

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

**Minimal supervised loop:**

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

**Extensions you should plan for:**

- **Regularization**: add L2 (or L1) to the **scalar loss** before `backward`, or apply **weight decay** inside the optimizer on each `step` (equivalent for L2 + SGD in simple cases). Ensure parameter gradients include \partial \text{reg} / \partial W.
- **Train vs validation**: each epoch, run a **validation pass** (no dropout noise; no grad) on a separate split to track `val_loss` / `val_acc`.
- **Early stopping**: keep the best snapshot by validation metric; if it does not improve for `**patience`** epochs, **stop** and reload best weights (see Section 13).

---

## 7. Testing Strategy (Catch2 + FakeIt)

- **Catch2 v3** for unit tests
- **FakeIt** (optional) for mocking when testing classes with dependencies
- Add validation for invalid inputs and dimension mismatches

### Tests per Implementation Step


| Step                   | Test file                            | What to verify                                                                                  |
| ---------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------- |
| Tensor                 | `test_tensor.cpp`                    | matmul result, transpose, shape, element-wise ops                                               |
| Linear layer           | `test_linear_layer.cpp`              | forward output shape, backward gradient shape; gradient check                                   |
| ReLU, Sigmoid, Softmax | `test_activation.cpp`                | ReLU: zeros stay zero, positives pass; Sigmoid: range (0,1); Softmax: outputs sum to 1          |
| MSE, CrossEntropy      | `test_loss.cpp`                      | MSE: regression; CrossEntropy: classification (with softmax)                                    |
| NeuralNetwork          | `test_neural_network.cpp`            | forward/backward shapes                                                                         |
| Optimizers             | `test_optimizer.cpp`                 | SGD/Momentum/Adam update formulas with known inputs                                             |
| Gradient check         | `test_gradient_check.cpp`            | Compare backward pass to numerical gradients (finite diff)                                      |
| **Conv2d**             | `test_conv2d_layer.cpp`              | output shape vs kernel/stride/padding; backward vs numeric grad (small random net)              |
| **MaxPool2d**          | `test_max_pool2d_layer.cpp`          | forward values; backward routes gradient to max indices only                                    |
| **L2/L1 penalty**      | `test_regularization.cpp` (optional) | total loss = task loss + \lambda·penalty; gradient check on tiny net                            |
| **Dropout**            | `test_dropout_layer.cpp`             | inverted dropout: train vs eval; backward only through kept units; gradient check in train mode |


---

## 8. Implementation Order

### Completed baseline

1. **Tensor/Matrix** – Eigen integration or minimal Tensor class (CPU) → `test_tensor.cpp`
2. **Linear layer** – forward + backward → `test_linear_layer.cpp`
3. **ReLU, Sigmoid, Softmax** – forward + backward for all; Softmax for classification output layer → `test_activation.cpp`
4. **MSE and CrossEntropy loss** – MSE for regression; CrossEntropy for classification (with softmax) → `test_loss.cpp`
5. **NeuralNetwork** – stack layers, forward, backward, parameter collection → `test_neural_network.cpp`
6. **SGD optimizer** – basic training loop → `test_optimizer.cpp`
7. **Demos** – MNIST and CIFAR-10 with MLP-style models
8. **GPU backend** – Tensor abstraction + cuBLAS + custom CUDA kernels for existing ops

### Next: CNN core (CPU first)

1. **4D tensor layout** – Agree on **NCHW** (batch, channels, height, width); helpers for computing output sizes *after conv/pool* (given kernel and stride).
2. **Conv2d** – forward + backward (reference: explicit loops or im2col + GEMM on CPU). **Initial implementation:** **no padding** (valid convolution), **stride 1** on height and width. **Future:** **stride > 1** (downsampling in the conv itself).
3. **MaxPool2d** – forward + backward with stored argmax
4. **Flatten** (or explicit reshape) – connect conv blocks to existing **Linear → Softmax**
5. **End-to-end CNN** – e.g. `Conv → ReLU → Pool` × … → `Flatten → Linear → Softmax`; train on **MNIST**, then **CIFAR-10**
6. **Dropout layer** – `dropout_layer.hpp` + `test_dropout_layer.cpp`; inverted dropout, **train** vs **eval**; place after **Linear** (and/or after flatten into dense blocks) as needed
7. **Gradient check** on a **tiny** CNN (same spirit as MLP finite-diff validation); include a run with dropout in **train** mode (fixed RNG seed)

### After CNN CPU

1. **Momentum / RMSprop / Adam** – if not already done
2. **CNN on GPU** – im2col/col2im + cuBLAS for conv; CUDA kernels for pool (or staged rollout); **dropout** on GPU = mask multiply + scale (same semantics as CPU)
3. **Runtime `Layer` + `SequentialModule` (polymorphic)** – `std::vector<std::shared_ptr<Layer>>`, virtual `forward`/`backward`, same wiring model as latches; this is the API surface for Python (see Section 10). Keep variadic `SequentialNN` for C++-only use if you still want it.
4. **Python bindings** – pybind11/nanobind against the **runtime** API (`Tensor`, `Layer` subclasses, `SequentialModule`, `Optimizer`, CNN layers, **Dropout**)
5. **Model save/load** – `save(path)` and `load(path)` for networks that include conv layers

### Generalization and efficiency (after core training works)

1. **L2 and/or L1 regularization** – add to total loss or via optimizer `weight_decay`; verify gradients on a toy net
2. **Train/validation split + early stopping** – track validation metric, `patience`, restore best weights (validation runs with **dropout off**)
3. **Smaller footprints** – favor narrower/fewer layers in experiments; optional later: magnitude pruning, INT8/FP16 storage (document in THEORY when you tackle it)
4. **CIFAR-10 strong baselines** – BatchNorm after convs, train-only augmentation (random crop + flip), LR schedule (cosine or step) + long runs; see **§12 — Strong CIFAR-10 training recipe (planned)**

---

## 9. Validation and Model Persistence

- **XOR problem**: 2-4-1 network with sigmoid, MSE or BCE. Should converge in hundreds of epochs.
- **Classification (e.g. MNIST)**: Final layer Linear → Softmax, loss CrossEntropy; outputs are class probabilities.
- **CNN (MNIST / CIFAR-10)**: Expect **better** generalization than a comparable MLP on images; use the same loaders you already have.
- **Linear regression**: 1-1 network, verify it matches analytical solution.
- **Gradient check**: Compare your backward pass to numerical gradients (finite differences) for small nets and small convs.
- **Model save/load**: Implement `save(path)` and `load(path)` for `NeuralNetwork` (extend when conv weights exist).
- **Regularization / early stopping**: Validation should show **less gap** between train and val when regularization is tuned; early stopping should **not** improve best val by continuing forever—compare against a fixed epoch budget.

---

## 10. Python Bindings

**Why runtime layers matter:** Bindings should target a `**SequentialModule`-style** container built from `**std::shared_ptr<Layer>`**, not the variadic `SequentialNN<..., A, B, C>`. pybind11 can register a **base class** with a trampoline if you want Python subclasses later; each concrete C++ layer type gets a `py::class_<LinearLayer, Layer, std::shared_ptr<LinearLayer>>` (or equivalent) and a factory on the module (`Linear(...)`, `ReLU()`, …).

**Expose at minimum:** `Tensor`, abstract `Layer`, concrete layer types (including `**Dropout`** with `p` and train/eval), `**SequentialModule**` (or named builder), `Optimizer`, loss, and a small training helper. Add `pytest` tests in `python/tests/`.

**Optional later:** expose only **factories** (`module.add_linear(...)`) without subclassing from Python, if you want a thinner binding surface first.

---

## 11. GPU Implementation (cuBLAS + CUDA)

**Chosen backend**: **cuBLAS** for NVIDIA GPUs. Build with `-DUSE_GPU=ON`. cuBLAS handles GEMM; custom CUDA kernels handle ReLU, element-wise ops, sum.

**CNN extension**: Prefer **im2col** (or similar) + GEMM for Conv2d on GPU to reuse cuBLAS; implement **col2im** for the input gradient. Max-pool backward uses the stored argmax map and scatter-add into `d_input`.

---

## 12. Convolutional Neural Networks — Current Phase (Detail)

**Goal:** Image-friendly models with local filters and translation-aware features, using the same manual-backprop discipline as your MLP.

**Tensor layout:** **NCHW** — batch `N`, channels `C`, height `H`, width `W`. Document this in code and tests so layers stay consistent.

**Minimal layer set:**


| Layer                      | Role                                                                                                   |
| -------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Conv2d**                 | Learned filters; stride and padding explicit in API                                                    |
| **ReLU**                   | Reuse your existing activation on conv outputs (4D elementwise)                                        |
| **MaxPool2d**              | Downsample; backward via argmax routing                                                                |
| **Flatten**                | Reshape to 2D for the classifier head                                                                  |
| **Linear**                 | Existing layer                                                                                         |
| **Dropout**                | After dense **Linear** blocks (and optionally on flattened vectors); **off** at inference / validation |
| **Softmax + CrossEntropy** | Existing loss/head                                                                                     |


**Typical CIFAR-style stack (example):** `Conv → ReLU → Pool` … → `Flatten → Linear → **Dropout** → Linear → Softmax` (exact depth is up to you).

**Implementation strategy:**

- **CPU path first**: Correctness and tests before performance. Options: naive nested loops for clarity, or **im2col + single GEMM** per forward/backward step to match how you will accelerate later.
- **Backward**: Derive or reference the conv gradient; implement `backward` and lock it in with **finite-difference checks** on small shapes.
- **Demos**: Swap the MLP trunk for a small CNN on MNIST, then scale architecture modestly for CIFAR-10.

**Success criteria:**

- Tests pass for conv and pool (including gradient check on small cases).
- Training loss decreases on MNIST with a simple CNN; CIFAR-10 shows sensible learning vs your old MLP baseline (exact accuracy target is optional—stability matters first).

### Strong CIFAR-10 training recipe (planned)

To move from “learning curves look good” toward **competitive test accuracy** (e.g. **~90%+** on CIFAR-10), plan these **in addition to** a sufficiently deep/wide conv backbone (e.g. ResNet-style blocks are the usual reference):

| Piece | Role |
| ----- | ---- |
| **Batch normalization** | After conv layers (and often before activation, depending on convention): stabilizes optimization, allows higher learning rates, improves accuracy. Requires **train vs eval** behavior (running mean/var updated in training; fixed stats at inference). |
| **Data augmentation (train only)** | **Random crop** to 32×32 (with padding) + **horizontal flip** are the standard minimum; reduces overfitting and improves generalization. Optional later: Cutout, stronger policies. |
| **Learning rate schedule** | **Cosine decay** or **step decay** of LR; pair with **many epochs** (often 200–350 in published recipes). Constant LR alone often plateaus below what a schedule can reach. |
| **Weight decay** | L2 on weights (often used together with BN + schedule); add to planning alongside existing regularization notes (Section 13). |

**Expectation:** A tiny 2-block CNN may still sit in the **~70–80%** range without augmentation and normalization; **>90%** usually assumes **BatchNorm + aug + schedule + enough capacity** (and often longer training), not hyperparameter tweaks alone.

---

## 13. Model efficiency, early stopping, and regularization

**Goal:** Train models that **generalize** and can be **kept small** where possible—without abandoning your manual-backprop style.

### Reducing model size


| Approach                 | Role                                                                                                 |
| ------------------------ | ---------------------------------------------------------------------------------------------------- |
| **Architecture choices** | Fewer layers, smaller `hidden_dim`, fewer conv channels—first and clearest lever                     |
| **Regularization**       | Often allows a smaller net to match a larger unregularized one on validation                         |
| **Pruning** (later)      | Remove weights/filters below a threshold; optional fine-tune                                         |
| **Quantization** (later) | Store activations/weights in fewer bits; training still often FP32 until you add specialized kernels |


Start with **narrower CNNs + L2 + early stopping** before advanced compression.

### Early stopping

- Hold out a **validation set** (or k-fold; simplest is single split).
- Each epoch: compute **validation loss** (and optionally accuracy) with **inference mode** (e.g. dropout off).
- Track **best** metric; if it does not improve for `**patience`** consecutive epochs, **stop** training.
- **Restore** weights from the best epoch (keep a copy in memory or reload from checkpoint).

This lives in the **training driver** (demo or `training/` helpers), not inside a single tensor op.

### Regularization terms

- **L2**: \lambda \sum_i w_i^2 added to loss; derivative adds 2\lambda w_i to each weight’s gradient (sign conventions vary—pick one and stay consistent).
- **L1**: encourages sparsity; gradient is \lambda \operatorname{sign}(w) where defined; implement carefully at w=0.

**Combined objective:** `total_loss = task_loss + l2_term + ...`; run `backward` on the sum so all terms participate in the graph.

### Dropout (layer)

Implement **Dropout** as a first-class **layer** (not only a bullet under “regularization”):

- **Inverted dropout (recommended):** training forward y = (x \odot m) / p with m_i \sim \mathrm{Bernoulli}(p); **evaluation** forward y = x (no scaling). Backward in training: propagate `grad_output` only through surviving units and divide by p.
- **State:** store the **mask** from the last forward for backward; thread a `**training`** (or `eval`) flag from the training loop so validation/inference **skip** masking.
- **Shapes:** support the same layouts as your activations (**2D** MLP activations and **4D** NCHW after conv if you choose—often dropout on dense layers first).
- **GPU:** elementwise multiply + optional RNG kernel; semantics match CPU.

**Bindings note:** Expose `Dropout(p)`, `model.train()` / `model.eval()` (or per-forward flag), regularization coefficients, and early-stopping options on your **training helper** or `TrainConfig` so Python can tune them without recompiling.