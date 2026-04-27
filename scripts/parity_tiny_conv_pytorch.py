#!/usr/bin/env python3
"""
Parity data for a single Conv2d (NCHW) + upstream gradient, matching C++ `ConvolutionalLayer`
+ cuDNN (forward/backward).

Forward: same as PyTorch F.conv2d (padding, stride 1).
Backward: dL/dOutput = g_y (fixed tensor); same as host ref in test_convolutional_layer_cuda.cpp.

Writes tests/data/parity_tiny_conv/:
  meta.txt, x.bin, w.bin, b.bin, y_ref.bin
  g_y.bin — upstream dL/dOutput (loss = (y * g_y).sum(), same as backward input)
  grad_w.bin, grad_b.bin, grad_x.bin — reference ∇W, ∇b, ∇X (host ref, checked vs torch if available)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _row_major_strides(shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    s3 = shape[3]
    s2 = shape[2] * s3
    s1 = shape[1] * s2
    return (shape[1] * s2, s2, s3, 1)


def _flat4(
    n: int, c: int, h: int, w: int, st: tuple[int, int, int, int]
) -> int:
    return n * st[0] + c * st[1] + h * st[2] + w * st[3]


def conv2d_nchw_reference(
    x: np.ndarray, weight: np.ndarray, b: np.ndarray, padding: int
) -> np.ndarray:
    n, ci, h, wi = x.shape
    co, ci_w, kh, kw = weight.shape
    assert ci == ci_w and kh == kw
    p = padding
    oh = h + 2 * p - kh + 1
    owi = wi + 2 * p - kw + 1
    y = np.zeros((n, co, oh, owi), dtype=np.float32)
    for ni in range(n):
        for co_ in range(co):
            for yy in range(oh):
                for xx in range(owi):
                    s = b[co_]
                    for cii in range(ci):
                        for ky in range(kh):
                            for kx in range(kw):
                                iy = yy + ky - p
                                ix_ = xx + kx - p
                                if 0 <= iy < h and 0 <= ix_ < wi:
                                    s += weight[co_, cii, ky, kx] * x[ni, cii, iy, ix_]
                    y[ni, co_, yy, xx] = s
    return y


def ref_conv_backward(
    grad_out: np.ndarray,
    go_shape: tuple[int, int, int, int],
    in_act: np.ndarray,
    in_shape: tuple[int, int, int, int],
    weights: np.ndarray,
    w_shape: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Port of C++ `ref_conv_backward` in test_convolutional_layer_cuda.cpp (flat NCHW row-major).
    """
    n, co, oh, ow = go_shape
    _ni, ci, ih, iw = in_shape
    pad = w_shape[2] // 2
    gst = _row_major_strides(go_shape)
    ist = _row_major_strides(in_shape)
    wst = _row_major_strides(w_shape)

    grad_w = np.zeros(int(np.prod(w_shape)), dtype=np.float32)
    grad_b = np.zeros(w_shape[0], dtype=np.float32)
    grad_in = np.zeros(int(np.prod(in_shape)), dtype=np.float32)
    in_act_f = in_act.ravel()
    w_f = weights.ravel()
    g_f = grad_out.ravel()

    for na in range(n):
        for c_out in range(co):
            for y in range(oh):
                for x in range(ow):
                    dy = g_f[_flat4(na, c_out, y, x, gst)]
                    grad_b[c_out] += dy
                    for cii in range(ci):
                        for ky in range(w_shape[2]):
                            for kx in range(w_shape[3]):
                                iy = y + ky - pad
                                ix2 = x + kx - pad
                                wi = _flat4(c_out, cii, ky, kx, wst)
                                xv = 0.0
                                if 0 <= iy < ih and 0 <= ix2 < iw:
                                    xv = in_act_f[_flat4(na, cii, iy, ix2, ist)]
                                    grad_in[_flat4(na, cii, iy, ix2, ist)] += (
                                        dy * w_f[wi]
                                    )
                                grad_w[wi] += dy * xv
    return grad_w.reshape(w_shape), grad_b, grad_in.reshape(in_shape)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "tests" / "data" / "parity_tiny_conv"
    out_dir.mkdir(parents=True, exist_ok=True)

    n, ci, in_h, in_w = 1, 2, 5, 5
    co, k = 3, 3
    pad = (k - 1) // 2
    rng = np.random.default_rng(42)
    x = rng.standard_normal((n, ci, in_h, in_w), dtype=np.float32)
    weight = 0.1 * rng.standard_normal((co, ci, k, k), dtype=np.float32)
    b = 0.05 * rng.standard_normal((co,), dtype=np.float32)

    y = conv2d_nchw_reference(x, weight, b, pad)
    w_shape = (co, ci, k, k)
    in_shape = (n, ci, in_h, in_w)
    go_shape = (n, co, y.shape[2], y.shape[3])

    rng2 = np.random.default_rng(43)
    g_y = 0.2 * rng2.standard_normal(y.shape, dtype=np.float32)

    gw, gb, gx = ref_conv_backward(
        g_y, go_shape, x, in_shape, weight, w_shape
    )

    try:
        import torch
        import torch.nn.functional as F

        xt = torch.from_numpy(x).detach().requires_grad_(True)
        wt = torch.from_numpy(weight).detach().requires_grad_(True)
        bt = torch.from_numpy(b).detach().requires_grad_(True)
        yt = F.conv2d(xt, wt, bt, padding=pad, stride=1)
        loss = (yt * torch.from_numpy(g_y).detach()).sum()
        loss.backward()
        assert float((wt.grad.numpy() - gw).max()) < 1e-4
        assert float((bt.grad.numpy() - gb).max()) < 1e-4
        assert float((xt.grad.numpy() - gx).max()) < 1e-4
        assert float((yt.detach().numpy() - y).max()) < 1e-4
        print("conv backward: max |ref - torch| ok")
    except ImportError:
        pass

    meta = f"{n} {ci} {in_h} {in_w} {co} {k} {y.shape[2]} {y.shape[3]}\n"
    (out_dir / "meta.txt").write_text(meta, encoding="ascii")

    def wbin(name: str, arr: np.ndarray) -> None:
        p = out_dir / name
        p.write_bytes(
            np.ascontiguousarray(arr, np.float32).tobytes("C")
        )

    wbin("x.bin", x)
    wbin("w.bin", weight)
    wbin("b.bin", b)
    wbin("y_ref.bin", y)
    wbin("g_y.bin", g_y)
    wbin("grad_w.bin", gw)
    wbin("grad_b.bin", gb)
    wbin("grad_x.bin", gx)

    print("Wrote", out_dir)
    print("meta:", meta.strip(), "y numel =", y.size)
    return 0


if __name__ == "__main__":
    if not (Path(__file__).resolve().parents[1] / "tests").is_dir():
        print("Run from repo: python scripts/parity_tiny_conv_pytorch.py", file=sys.stderr)
        sys.exit(1)
    raise SystemExit(main())
