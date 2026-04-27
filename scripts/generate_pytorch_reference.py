#!/usr/bin/env python3
"""
Regenerate *all* float32 reference blobs under tests/data/pytorch_ref/.  MLP and conv2d
require **PyTorch**; momentum SGD, softmax+CE, and their optional torch self-checks use **NumPy**
with the same numerics (PyTorch is verified when `import torch` succeeds).
C++ tests compare against the written `.bin` files.

  pip install torch numpy
  python scripts/generate_pytorch_reference.py

Subpackages:
  pytorch_ref/mlp/   — Linear(4,3) → ReLU → Linear(3,2), MSE backward
  pytorch_ref/conv2d/ — single Conv2d, scalar loss = sum(y * g_y) for backward
  pytorch_ref/momentum_sgd/ — Linear(2,1) + MSE(mean), 2 SGD+momentum(damp=0) steps (plain)
  pytorch_ref/momentum_sgd_nesterov/ — same with nesterov=True (optional torch self-check)
  pytorch_ref/softmax_cross_entropy/ — SoftmaxCrossEntropyLoss: logits+one-hot (or soft) targets, mean over batch; optional torch check

C++ weight layout: Linear (in, out) = W_pt.T; conv (Co,Ci,k,k) same as PyTorch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def require_torch():
    import torch

    return torch


def write_float32(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes("C"))


def gen_mlp(out_dir: Path) -> None:
    torch = require_torch()
    import torch.nn as nn
    import torch.nn.functional as F

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

    w0_cpp = w0_pt.T.copy()
    w1_cpp = w1_pt.T.copy()
    gw0_cpp = m[0].weight.grad.detach().cpu().numpy().T.copy()
    gw1_cpp = m[2].weight.grad.detach().cpu().numpy().T.copy()
    gb0 = m[0].bias.grad.detach().cpu().numpy()
    gb1 = m[2].bias.grad.detach().cpu().numpy()
    gx = x_t.grad.detach().cpu().numpy()
    y = y_t.detach().cpu().numpy()

    files = {
        "w0.bin": w0_cpp,
        "b0.bin": b0.reshape(3, 1),
        "w1.bin": w1_cpp,
        "b1.bin": b1.reshape(2, 1),
        "x.bin": x,
        "y_ref.bin": y,
        "target.bin": target,
        "grad_w0.bin": gw0_cpp,
        "grad_b0.bin": gb0.reshape(3, 1),
        "grad_w1.bin": gw1_cpp,
        "grad_b1.bin": gb1.reshape(2, 1),
        "grad_x.bin": gx,
    }
    for name, arr in files.items():
        write_float32(out_dir / name, arr)
    print("Wrote MLP ref under", out_dir)


def gen_conv2d(out_dir: Path) -> None:
    torch = require_torch()
    import torch.nn.functional as F

    out_dir.mkdir(parents=True, exist_ok=True)

    n, ci, in_h, in_w = 1, 2, 5, 5
    co, k = 3, 3
    pad = (k - 1) // 2
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((n, ci, in_h, in_w), dtype=np.float32)
    weight_np = 0.1 * rng.standard_normal((co, ci, k, k), dtype=np.float32)
    b_np = 0.05 * rng.standard_normal((co,), dtype=np.float32)

    x = torch.from_numpy(x_np).requires_grad_(True)
    w = torch.from_numpy(weight_np).requires_grad_(True)
    b = torch.from_numpy(b_np).requires_grad_(True)
    y = F.conv2d(x, w, b, padding=pad, stride=1)

    rng2 = np.random.default_rng(43)
    g_y_np = 0.2 * rng2.standard_normal(
        (n, co, y.shape[2], y.shape[3]), dtype=np.float32
    )
    g_y = torch.from_numpy(g_y_np)
    loss = (y * g_y).sum()
    loss.backward()

    grad_w = w.grad.detach().cpu().numpy()
    grad_b = b.grad.detach().cpu().numpy()
    grad_x = x.grad.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    meta = f"{n} {ci} {in_h} {in_w} {co} {k} {y_np.shape[2]} {y_np.shape[3]}\n"
    (out_dir / "meta.txt").write_text(meta, encoding="ascii")
    write_float32(out_dir / "x.bin", x_np)
    write_float32(out_dir / "w.bin", weight_np)
    write_float32(out_dir / "b.bin", b_np)
    write_float32(out_dir / "y_ref.bin", y_np)
    write_float32(out_dir / "g_y.bin", g_y_np)
    write_float32(out_dir / "grad_w.bin", grad_w)
    write_float32(out_dir / "grad_b.bin", grad_b)
    write_float32(out_dir / "grad_x.bin", grad_x)
    print("Wrote conv2d ref under", out_dir, "meta:", meta.strip())


def _momentum_sgd_pytorch_ref_numpy( nesterov: bool ):
    """
    MSE(mean) on batch 4, Linear(2,1) in C++ layout: w (2,1), b (1,1).
    Two SGD + momentum(dampening=0) steps matching torch.optim.SGD
    (see PyTorch sgd.py: nesterov uses g + mu * buf with buf = mu*buf + g).
    """
    w = np.array([[0.1], [-0.2]], dtype=np.float32)
    b = np.array([[0.3]], dtype=np.float32)
    w_init = w.copy()
    b_init = b.copy()
    x = np.array(
        [[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75]], dtype=np.float32
    )
    t = np.array([[0.0], [0.125], [0.25], [0.375]], dtype=np.float32)
    lr, mom, wd, n_steps = 0.02, 0.9, 0.0, 2
    buf_w, buf_b = None, None
    n_el = 4.0
    for _ in range( n_steps ):
        y = x @ w + b
        diff = y - t
        d_y = (2.0 / n_el) * diff
        gw = x.T @ d_y
        gb = np.sum( d_y, axis=0, keepdims=True ).astype( np.float32 )
        d_p_w = np.ascontiguousarray( gw ) + wd * w
        d_p_b = np.ascontiguousarray( gb ) + wd * b
        d0_w, d0_b = d_p_w.copy(), d_p_b.copy()
        if buf_w is None:
            buf_w = d0_w.copy()
        else:
            buf_w = mom * buf_w + d0_w
        if buf_b is None:
            buf_b = d0_b.copy()
        else:
            buf_b = mom * buf_b + d0_b
        if nesterov:
            up_w = d0_w + mom * buf_w
            up_b = d0_b + mom * buf_b
        else:
            up_w, up_b = buf_w, buf_b
        w = w - lr * up_w
        b = b - lr * up_b
    return w_init, b_init, w, b, x, t, lr, mom, n_steps


def gen_momentum_sgd( out_dir: Path, nesterov: bool ) -> None:
    out_dir.mkdir( parents=True, exist_ok=True )
    w0, b0, wf, bf, x, t, lr, mom, n_steps = _momentum_sgd_pytorch_ref_numpy( nesterov )
    ne = 1 if nesterov else 0
    # Optional self-check with PyTorch on the same setup
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        w_pt = w0.T.copy()
        m = nn.Linear( 2, 1, bias=True )
        m.weight.data.copy_( torch.from_numpy( w_pt ) )
        m.bias.data.copy_( torch.from_numpy( b0.reshape( 1 ) ) )
        opt = torch.optim.SGD(
            m.parameters(),
            lr=lr,
            momentum=mom,
            weight_decay=0.0,
            nesterov=bool( nesterov ),
        )
        for _ in range( n_steps ):
            opt.zero_grad()
            y = m( torch.from_numpy( x ) )
            F.mse_loss( y, torch.from_numpy( t ), reduction="mean" ).backward()
            opt.step()
        w_t = m.weight.data.detach().cpu().numpy().T
        b_t = m.bias.data.detach().cpu().numpy().reshape( 1, 1 )
        if not (
            np.allclose( w_t, wf, atol=1e-5, rtol=1e-5 )
            and np.allclose( b_t, bf, atol=1e-5, rtol=1e-5 )
        ):
            raise RuntimeError( "NumPy SGD ref vs torch mismatch" )
    except ImportError:
        pass

    write_float32( out_dir / "w_init.bin", w0 )
    write_float32( out_dir / "b_init.bin", b0 )
    write_float32( out_dir / "w_after.bin", wf )
    write_float32( out_dir / "b_after.bin", bf )
    write_float32( out_dir / "x.bin", x )
    write_float32( out_dir / "t.bin", t )
    meta = f"in_f=2 out_f=1 n_rows=4 batch=4 n_steps={n_steps} lr={lr} mom={mom} l2=0.0 nesterov={ne}\n"
    ( out_dir / "meta.txt" ).write_text( meta, encoding="ascii" )
    print( "Wrote momentum_sgd", "nesterov" if nesterov else "plain", "under", out_dir )


def gen_softmax_cross_entropy( out_dir: Path ) -> None:
    """
    Matches `SoftmaxCrossEntropyLoss` in include/losses/cross_entropy_softmax_loss.hpp
    (stable softmax, loss = (1/B) * asum(t ⊙ log p), backward (p - t) / B).
    """
    out_dir.mkdir( parents=True, exist_ok=True )
    rng = np.random.default_rng( 2027 )
    B, C = 3, 5
    logits = ( 0.5 * rng.standard_normal( ( B, C ), dtype=np.float32 ) ).astype( np.float32 )
    t = np.zeros( ( B, C ), dtype=np.float32 )
    for b in range( B ):
        t[b, ( b * 2 ) % C] = 1.0

    row_max = np.max( logits, axis=1, keepdims=True )
    m_probs = np.exp( logits - row_max )
    s = m_probs.sum( axis=1, keepdims=True )
    p = m_probs / s
    logp = np.log( p )
    weighted = t * logp
    m_inv = 1.0 / float( B )
    loss_v = ( np.sum( np.abs( weighted ) ) * m_inv ).astype( np.float32 )
    grad = ( p - t ) * m_inv
    try:
        import torch
        import torch.nn.functional as F

        Lth = torch.from_numpy( logits )
        Tt = torch.from_numpy( t )
        loss2 = ( -Tt * F.log_softmax( Lth, 1 ) ).sum() * ( 1.0 / B )
        if not bool(
            np.isclose( float( loss2.item() ), float( loss_v ), rtol=1e-4, atol=1e-5 )
        ):
            raise RuntimeError( "softmax CE loss vs torch" )
        Lth2 = Lth.clone().requires_grad_( True )
        l3 = ( -Tt * F.log_softmax( Lth2, 1 ) ).sum() * ( 1.0 / B )
        l3.backward()
        gth = Lth2.grad.detach().cpu().numpy()
        if not np.allclose( gth, grad, rtol=1e-4, atol=1e-5 ):
            raise RuntimeError( "softmax CE grad vs torch" )
    except ImportError:
        pass

    write_float32( out_dir / "logits.bin", logits )
    write_float32( out_dir / "target.bin", t )
    write_float32( out_dir / "loss.bin", np.array( [ loss_v ], dtype=np.float32 ) )
    write_float32( out_dir / "grad_logits.bin", grad.astype( np.float32 ) )
    meta = f"B={B} C={C}\n"
    ( out_dir / "meta.txt" ).write_text( meta, encoding="ascii" )
    print( "Wrote softmax_cross_entropy ref under", out_dir )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate tests/data/pytorch_ref/* from PyTorch (required)."
    )
    ap.add_argument(
        "what",
        nargs="?",
        default="all",
        choices=("all", "mlp", "conv2d", "momentum", "momentum_nesterov", "softmax_ce"),
        help="which reference pack to build (default: all)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    if not (root / "tests").is_dir():
        print("Run from repository root.", file=sys.stderr)
        return 1

    base = root / "tests" / "data" / "pytorch_ref"
    if args.what in ("all", "mlp"):
        gen_mlp(base / "mlp")
    if args.what in ("all", "conv2d"):
        gen_conv2d(base / "conv2d")
    if args.what in ("all", "momentum"):
        gen_momentum_sgd(base / "momentum_sgd", nesterov=False)
    if args.what in ("all", "momentum_nesterov"):
        gen_momentum_sgd(base / "momentum_sgd_nesterov", nesterov=True)
    if args.what in ("all", "softmax_ce"):
        gen_softmax_cross_entropy( base / "softmax_cross_entropy" )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImportError as e:
        print("PyTorch is required: pip install torch", file=sys.stderr)
        print(e, file=sys.stderr)
        raise SystemExit(1) from e
