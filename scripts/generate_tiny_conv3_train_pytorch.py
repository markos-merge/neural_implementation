#!/usr/bin/env python3
"""
Build reference for C++ test: 2 images (3,8,8) → Conv→BN→ReLU→Dropout(p=0)→Linear(512,4),
softmax+CE in mean-one-hot form matching neural::SoftmaxCrossEntropyLoss.

3 consecutive SGD(lr) steps on the *same* batch of 2 samples; logs loss (before opt.step) each step,
then logits in eval() after 3 steps.

  pip install torch numpy
  python scripts/generate_tiny_conv3_train_pytorch.py

Outputs tests/data/pytorch_ref/tiny_conv3_train/ (float32 row-major, same host layout
used to construct neural::CudaTensor4 / CudaTensor in CUDA tests)

  - x_nchw.bin  — (B,C,H,W) NCHW, C-contiguous, for CudaTensor4 input
  - x_flat.bin  — (B, C*H*W) rows (duplicate view of the same data)
  - After 3 train steps: bn_running_mean.bin, bn_running_var.bin — (1, Co, 1, 1) for eval
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    print("PyTorch is required: pip install torch", file=sys.stderr)
    raise SystemExit(1) from e


def write_f32(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes("C"))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out = root / "tests" / "data" / "pytorch_ref" / "tiny_conv3_train"
    out.mkdir(parents=True, exist_ok=True)

    B, C, H, W, K, Co, n_cls = 2, 3, 8, 8, 3, 8, 4
    flat = Co * H * W
    assert flat == 512

    torch.manual_seed(2028)
    np.random.seed(2028)
    rng = np.random.default_rng(2028)
    # NCHW row-major in x.bin matches flat rows (C*H*W) per image
    x_nchw = (0.1 * rng.standard_normal((B, C, H, W), dtype=np.float32)).astype(np.float32)
    t_onehot = np.zeros((B, n_cls), dtype=np.float32)
    t_onehot[0, 0] = 1.0
    t_onehot[1, 2] = 1.0

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(C, Co, K, padding=1, bias=True)
            self.bn = nn.BatchNorm2d(Co, eps=1e-5, momentum=0.1)
            self.drop = nn.Dropout2d(0.0)  # p=0: identity, matches C++ keep_prob=1
            self.lin = nn.Linear(flat, n_cls)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.conv(x)
            h = F.relu(self.bn(h))
            h = self.drop(h)
            h = h.view(B, -1)
            return self.lin(h)

    net = Net()
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
            nn.init.xavier_uniform_(m.weight, gain=0.8)
            nn.init.zeros_(m.bias)
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.0, weight_decay=0.0)

    x_t = torch.from_numpy(x_nchw)
    t_t = torch.from_numpy(t_onehot)

    # one-hot mean CE matching C++: -(t * log_softmax) / B
    def ce(tgt: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return (-tgt * F.log_softmax(logits, 1)).sum() / (float(B) * 1.0)

    x_flat = x_nchw.reshape(B, C * H * W)
    write_f32(out / "x_nchw.bin", x_nchw)
    write_f32(out / "x_flat.bin", x_flat)
    write_f32(out / "t_onehot.bin", t_onehot)

    w_conv = net.conv.weight.detach().cpu().numpy()  # (Co,Ci,K,K)
    b_conv = net.conv.bias.detach().cpu().numpy().reshape(1, Co, 1, 1)
    write_f32(out / "conv_w.bin", w_conv)
    write_f32(out / "conv_b.bin", b_conv)
    g = net.bn.weight.detach().cpu().numpy()
    b = net.bn.bias.detach().cpu().numpy()
    write_f32(out / "bn_gamma.bin", g.reshape(1, Co, 1, 1))
    write_f32(out / "bn_beta.bin", b.reshape(1, Co, 1, 1))
    w_pt = net.lin.weight.detach().cpu().numpy()  # (n_cls, flat)
    write_f32(out / "lin_w_cpp.bin", w_pt.T.copy())  # (flat, n_cls) row-major
    b_lin = net.lin.bias.detach().cpu().numpy()
    write_f32(out / "lin_b.bin", b_lin.reshape(n_cls, 1))

    losses: list[float] = []
    net.train()
    for _ in range(3):
        opt.zero_grad()
        logits = net(x_t)
        lo = ce(t_t, logits)
        lo.backward()
        losses.append(float(lo.item()))
        opt.step()

    write_f32(out / "losses3.bin", np.array(losses, dtype=np.float32))
    # Running stats (train-mode forwards); needed to mirror eval with fixed BN in C++ tests.
    rm = net.bn.running_mean.detach().cpu().numpy()
    rv = net.bn.running_var.detach().cpu().numpy()
    write_f32(out / "bn_running_mean.bin", rm.reshape(1, Co, 1, 1))
    write_f32(out / "bn_running_var.bin", rv.reshape(1, Co, 1, 1))

    net.eval()
    with torch.no_grad():
        logits_e = net(x_t).detach().cpu().numpy()
    write_f32(out / "logits_eval.bin", logits_e)

    meta = f"B={B} C={C} H={H} W={W} K={K} Co={Co} n_cls={n_cls} flat={flat} lr=0.05 sgd_steps=3\n"
    (out / "meta.txt").write_text(meta, encoding="ascii")
    print("Wrote", out, "losses", losses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
