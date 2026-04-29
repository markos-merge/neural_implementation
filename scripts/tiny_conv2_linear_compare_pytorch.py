#!/usr/bin/env python3
"""
PyTorch twin of `tiny_conv2_linear_compare` (two 3×3 conv + ReLU, then one linear).

Loads weights/input from the C++ binary, checks forward logits, then runs one training step:
`CrossEntropyLoss` (mean) matching `SoftmaxCrossEntropyLoss`, compares gradients to C++ dumps,
applies SGD (`lr`, `weight_decay` matching `sgd_optimizer.hpp`), compares updated weights.

Usage (after building with CUDA + cuDNN):
  ./build_release_cuda/tiny_conv2_linear_compare ./tiny_compare_artifacts
  conda activate torch-clean
  python scripts/tiny_conv2_linear_compare_pytorch.py --artifact-dir ./tiny_compare_artifacts

Optional: pass `--cpp-exe` to run the C++ binary first.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Must match src/app/tiny_conv2_linear_compare.cpp
C, H, W = 3, 8, 8
CONV1_OUT, CONV2_OUT = 4, 2
LINEAR_OUT = 5
FLAT = CONV2_OUT * H * W
TARGET_CLASS = 2
LR = 0.01
WEIGHT_DECAY = 0.0


def load_f32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.reshape(-1) - b.reshape(-1))))


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(C, CONV1_OUT, 3, padding=1, bias=True)
        self.r1 = nn.ReLU()
        self.c2 = nn.Conv2d(CONV1_OUT, CONV2_OUT, 3, padding=1, bias=True)
        self.r2 = nn.ReLU()
        self.fc = nn.Linear(FLAT, LINEAR_OUT, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.r1(self.c1(x))
        x = self.r2(self.c2(x))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


def apply_weights_from_artifacts(m: TinyNet, d: Path) -> None:
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            nn.init.zeros_(mod.bias)

    w1 = load_f32(d / "conv1_w.f32").reshape(CONV1_OUT, C, 3, 3)
    w2 = load_f32(d / "conv2_w.f32").reshape(CONV2_OUT, CONV1_OUT, 3, 3)
    lin = load_f32(d / "linear_w.f32").reshape(FLAT, LINEAR_OUT)

    with torch.no_grad():
        m.c1.weight.copy_(torch.from_numpy(w1))
        m.c2.weight.copy_(torch.from_numpy(w2))
        m.fc.weight.copy_(torch.from_numpy(lin).t().contiguous())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("./tiny_compare_artifacts"),
        help="Directory with *.f32 from the C++ compare binary",
    )
    p.add_argument(
        "--cpp-exe",
        type=Path,
        default=None,
        help="If set, run `<cpp-exe> <artifact-dir>` before loading artifacts",
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    a = p.parse_args()
    d = a.artifact_dir.resolve()

    if a.cpp_exe is not None:
        subprocess.run([str(a.cpp_exe), str(d)], check=True)

    req = [
        "input.f32",
        "conv1_w.f32",
        "conv2_w.f32",
        "linear_w.f32",
        "logits_cpp.f32",
        "train_loss_cpp.f32",
        "grad_conv1_w.f32",
        "grad_conv1_b.f32",
        "grad_conv2_w.f32",
        "grad_conv2_b.f32",
        "grad_linear_w.f32",
        "grad_linear_b.f32",
        "conv1_w_after.f32",
        "conv2_w_after.f32",
        "linear_w_after.f32",
    ]
    for name in req:
        if not (d / name).is_file():
            print(f"Missing {d / name}; run the C++ binary first or pass --cpp-exe.", file=sys.stderr)
            return 1

    x_np = load_f32(d / "input.f32").reshape(1, C, H, W)
    ref_logits = load_f32(d / "logits_cpp.f32")
    ref_loss = float(load_f32(d / "train_loss_cpp.f32")[0])
    g1w = load_f32(d / "grad_conv1_w.f32").reshape(CONV1_OUT, C, 3, 3)
    g1b = load_f32(d / "grad_conv1_b.f32").reshape(CONV1_OUT)
    g2w = load_f32(d / "grad_conv2_w.f32").reshape(CONV2_OUT, CONV1_OUT, 3, 3)
    g2b = load_f32(d / "grad_conv2_b.f32").reshape(CONV2_OUT)
    glw = load_f32(d / "grad_linear_w.f32").reshape(FLAT, LINEAR_OUT)
    glb = load_f32(d / "grad_linear_b.f32").reshape(LINEAR_OUT)
    w1_after = load_f32(d / "conv1_w_after.f32").reshape(CONV1_OUT, C, 3, 3)
    w2_after = load_f32(d / "conv2_w_after.f32").reshape(CONV2_OUT, CONV1_OUT, 3, 3)
    lin_after = load_f32(d / "linear_w_after.f32").reshape(FLAT, LINEAR_OUT)

    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    m = TinyNet().to(device)
    apply_weights_from_artifacts(m, d)

    x = torch.from_numpy(x_np).to(device)
    # --- forward (inference) vs C++ pre-train logits
    m.eval()
    with torch.inference_mode():
        y_inf = m(x).detach().cpu().numpy().reshape(-1)
    err_logits = max_abs(y_inf, ref_logits)
    print(f"device={device}  forward_max_abs_err_vs_cpp={err_logits:.6e}")

    # --- one optimization step vs C++
    m.train()
    y = m(x)
    tgt = torch.tensor([TARGET_CLASS], dtype=torch.long, device=device)
    crit = nn.CrossEntropyLoss(reduction="mean")
    loss = crit(y, tgt)
    loss.backward()

    err_loss = abs(float(loss.item()) - ref_loss)
    print(f"  train_loss_abs_err_vs_cpp={err_loss:.6e}  (py={loss.item():.9f} cpp={ref_loss:.9f})")

    assert m.c1.weight.grad is not None
    assert m.fc.weight.grad is not None
    eg1w = max_abs(m.c1.weight.grad.detach().cpu().numpy(), g1w)
    eg1b = max_abs(m.c1.bias.grad.detach().cpu().numpy(), g1b)
    eg2w = max_abs(m.c2.weight.grad.detach().cpu().numpy(), g2w)
    eg2b = max_abs(m.c2.bias.grad.detach().cpu().numpy(), g2b)
    eg_lw = max_abs(m.fc.weight.grad.detach().cpu().numpy().T, glw)
    eg_lb = max_abs(m.fc.bias.grad.detach().cpu().numpy(), glb)
    print(
        f"  grad_max_abs_err  conv1_w={eg1w:.6e} conv1_b={eg1b:.6e} "
        f"conv2_w={eg2w:.6e} conv2_b={eg2b:.6e} linear_w={eg_lw:.6e} linear_b={eg_lb:.6e}"
    )

    opt = torch.optim.SGD(m.parameters(), lr=LR, momentum=0.0, weight_decay=WEIGHT_DECAY)
    opt.step()

    ew1 = max_abs(m.c1.weight.detach().cpu().numpy(), w1_after)
    ew2 = max_abs(m.c2.weight.detach().cpu().numpy(), w2_after)
    ew_l = max_abs(m.fc.weight.detach().cpu().numpy().T, lin_after)
    print(f"  weights_after_step_max_abs_err  conv1={ew1:.6e} conv2={ew2:.6e} linear={ew_l:.6e}")

    tol_fwd = 5e-4
    tol_loss = 5e-5
    tol_g = 2e-3
    tol_w = 2e-3
    ok = (
        err_logits <= tol_fwd
        and err_loss <= tol_loss
        and eg1w <= tol_g
        and eg1b <= tol_g
        and eg2w <= tol_g
        and eg2b <= tol_g
        and eg_lw <= tol_g
        and eg_lb <= tol_g
        and ew1 <= tol_w
        and ew2 <= tol_w
        and ew_l <= tol_w
    )
    if not ok:
        print("  One or more tolerances exceeded (see thresholds in script).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
