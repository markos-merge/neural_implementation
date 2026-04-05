# Documentation index

Use this file to find **where things live**—theory, citations, datasets, and (when you add them) training artifacts.

## Theory write-ups

| Topic | Path |
|--------|------|
| Optimizers (SGD, momentum, RMSprop, Adam) | [`4.optimizers/THEORY.md`](4.optimizers/THEORY.md) |
| Tensors / layouts | [`1.tensor/`](1.tensor/) |
| Linear layer | [`2.linear_layer/THEORY.md`](2.linear_layer/THEORY.md) |
| Activations | [`3.activation/THEORY.md`](3.activation/THEORY.md) |
| Backprop walkthrough | [`BACKPROP_WALKTHROUGH.md`](BACKPROP_WALKTHROUGH.md) |
| Project plan (features, roadmap) | [`PLAN.md`](PLAN.md) |

## Citations and external sources

- **[`SOURCES.md`](SOURCES.md)** — papers, notes, and links (including momentum / Polyak / Nesterov / Sutskever).

## Data on disk (datasets, not in git)

Large binaries stay **outside** version control. The repo expects a top-level **`data/`** directory (listed in `.gitignore`). Put MNIST / CIFAR-10 archives or extracted files there and point demos at that path.

## Data in memory (optimizer state)

Things like **momentum velocities** are not files: they are **tensors the same shape as each trainable parameter**, owned by the **optimizer** (or equivalent), initialized to zero at training start, and updated every step. That matches the “what to store” note in [`4.optimizers/THEORY.md`](4.optimizers/THEORY.md).

## Checkpoints and logs (future)

Saving weights / optimizer state to disk is **not** wired up yet; see **`PLAN.md`** (model save/load, early stopping). When you add it, use a dedicated folder (e.g. `checkpoints/` or `runs/`) and add it to `.gitignore` if it should stay local.
