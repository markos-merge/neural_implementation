#!/usr/bin/env python3
"""
Generate reference weights, I/O, and gradients for a tiny net so C++ can parity-check
the same numerics as PyTorch:

  Linear(4 -> 3) -> ReLU -> Linear(3 -> 2)
  MSE loss (mean over all elements) — matches neural::MSELoss.

nn.Linear: y = x @ W^T + b, W (out, in). C++ weights are (in, out) = W^T in w0.bin / w1.bin.

Backward: fixed target, reference grads in C++ layout (grad w same shape as w).

  python scripts/parity_tiny_pytorch.py

Writes tests/data/parity_tiny/*.bin, including for backward (MSE, fixed target):
  target.bin, grad_w0.bin, grad_b0.bin, grad_w1.bin, grad_b1.bin, grad_x.bin
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def mse_grad_dldy(
    y: np.ndarray, target: np.ndarray, n: int | None = None
) -> np.ndarray:
    """MSELoss: L = (1/n_elem) * sum (y-t)^2, dL/dy = (2/n_elem) * (y-t)."""
    if n is None:
        n = y.size
    return (2.0 / n) * (y - target)


def mlp_backward_mse_numerical(
    x: np.ndarray, w0_pt: np.ndarray, b0: np.ndarray, w1_pt: np.ndarray, b1: np.ndarray, target: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Analytical MSE backward, PyTorch (out, in) weights — returns np grads for w0_pt, b0, w1_pt, b1, x."""
    z0 = x @ w0_pt.T + b0
    h = np.maximum(z0, 0.0)
    y = h @ w1_pt.T + b1
    n = y.size
    dldy = mse_grad_dldy(y, target, n)

    dld_w1_pt = dldy.T @ h
    dld_b1 = dldy.sum(axis=0)

    dldh = dldy @ w1_pt
    dldz0 = dldh * (z0 > 0).astype(np.float32)
    dld_w0_pt = dldz0.T @ x
    dld_b0 = dldz0.sum(axis=0)
    dld_x = dldz0 @ w0_pt

    return dld_w0_pt, dld_b0, dld_w1_pt, dld_b1, dld_x


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "tests" / "data" / "parity_tiny"
    out_dir.mkdir(parents=True, exist_ok=True)

    w0_pt = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [-0.1, -0.2, -0.3, -0.4],
            [1.0, 0.0, 0.5, -0.5],
        ],
        dtype=np.float32,
    )
    b0 = np.array([0.01, -0.02, 0.03], dtype=np.float32)
    w1_pt = np.array(
        [
            [0.5, 1.0, -0.5],
            [-1.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    b1 = np.array([0.1, -0.1], dtype=np.float32)
    x = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-1.0, 0.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [0.5, -0.25],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    h = np.maximum(x @ w0_pt.T + b0, 0.0)
    y = h @ w1_pt.T + b1

    w0_cpp = w0_pt.T.copy()
    w1_cpp = w1_pt.T.copy()

    gw0_pt, gb0, gw1_pt, gb1, gx = mlp_backward_mse_numerical(x, w0_pt, b0, w1_pt, b1, target)
    gw0_cpp = gw0_pt.T
    gw1_cpp = gw1_pt.T

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        m = nn.Sequential(
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )
        m[0].weight.data.copy_(torch.from_numpy(w0_pt))
        m[0].bias.data.copy_(torch.from_numpy(b0))
        m[2].weight.data.copy_(torch.from_numpy(w1_pt))
        m[2].bias.data.copy_(torch.from_numpy(b1))
        x_t = torch.from_numpy(x).requires_grad_(True)
        y_t = m(x_t)
        loss = F.mse_loss(y_t, torch.from_numpy(target), reduction="mean")
        loss.backward()
        d = float(np.max(np.abs(m[0].weight.grad.cpu().numpy() - gw0_pt)))
        assert d < 1e-5, d
        d = float(np.max(np.abs(m[2].weight.grad.cpu().numpy() - gw1_pt)))
        assert d < 1e-5, d
        d = float(np.max(np.abs(m[0].bias.grad.cpu().numpy() - gb0)))
        assert d < 1e-5, d
        d = float(np.max(np.abs(m[2].bias.grad.cpu().numpy() - gb1)))
        assert d < 1e-5, d
        d = float(np.max(np.abs(x_t.grad.cpu().numpy() - gx)))
        assert d < 1e-5, d
        print("MSE backward: max |ref - torch| ok")
    except ImportError:
        pass

    for name, arr in [
        ("w0.bin", w0_cpp),
        ("b0.bin", b0.reshape(3, 1)),
        ("w1.bin", w1_cpp),
        ("b1.bin", b1.reshape(2, 1)),
        ("x.bin", x),
        ("y_ref.bin", y),
        ("target.bin", target),
        ("grad_w0.bin", gw0_cpp),
        ("grad_b0.bin", gb0.reshape(3, 1)),
        ("grad_w1.bin", gw1_cpp),
        ("grad_b1.bin", gb1.reshape(2, 1)),
        ("grad_x.bin", gx),
    ]:
        (out_dir / name).write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes(order="C"))

    print("Wrote", out_dir)
    print("y_ref =\n", y)
    return 0


if __name__ == "__main__":
    if not (Path(__file__).resolve().parents[1] / "tests").is_dir():
        print("Run from repo: python scripts/parity_tiny_pytorch.py", file=sys.stderr)
        sys.exit(1)
    raise SystemExit(main())
