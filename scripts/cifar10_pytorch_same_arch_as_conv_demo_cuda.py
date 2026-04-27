#!/usr/bin/env python3
"""
Train the same topology as src/app/conv_demo_cuda.cpp on CIFAR-10 for wall-clock
comparison with the C++ CUDA/cuDNN build.

Dependencies:
  pip install torch torchvision

Geometry matches the cuDNN path (valid conv, odd 3×3 kernels; last conv is 1×1):
  32x32 -> Conv3x3 -> 30x30 -> pool2 -> 15x15 -> Conv3x3 -> 13x13 -> pool2 -> 6x6
  -> Conv3x3 -> 4x4 -> Conv3x3 -> 2x2 -> Conv1x1 -> 2x2
  Flatten 256*2*2 = 1024 -> Linear 512 -> ReLU -> Linear 256 -> ReLU -> Linear 10, CrossEntropyLoss (mean).

Hyperparameters default to conv_demo_cuda.cpp (SGD lr=0.001, batch 256, weight decay 1e-4
on conv/linear *weights* and BN *weight* only; biases and BN bias have no decay — same pattern
as the C++ optimizer).

Note: BatchNorm running-stat EMA in PyTorch uses momentum=0.1 naming convention(running = 0.9*old + 0.1*batch). The C++ cuDNN path uses exponentialAverageFactor = 1 - 0.1,
which corresponds to the opposite blend order in cuDNN’s formula — numerics differ slightly
from PyTorch for BN EMA, but layer shapes and parameter tying intent match the demo.

Usage:
  python scripts/cifar10_pytorch_same_arch_as_conv_demo_cuda.py --epochs 1
  python scripts/cifar10_pytorch_same_arch_as_conv_demo_cuda.py --epochs 8000 --data-dir ./data
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


LR = 0.001  # conv_demo_cuda.cpp opt.m_learning_rate (smaller LR for deeper net)
BATCH_SIZE = 256  # conv_demo_cuda.cpp opt.m_batch_size
WEIGHT_DECAY = 1e-4
BN_EPS = 1e-5
BN_MOMENTUM = 0.1
EPOCHS_DEFAULT = 8000


class ConvDemoCudaArch(nn.Module):
    """Same module stack as ConvBox_t + three Linear layers in conv_demo_cuda.cpp."""

    def __init__(self) -> None:
        super().__init__()
        # Match C++ valid conv geometry (padding=0 for 3x3; 1x1 last conv).
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(32, eps=BN_EPS, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64, eps=BN_EPS, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(128, eps=BN_EPS, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(128, eps=BN_EPS, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256, eps=BN_EPS, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


def he_init_relu(module: nn.Module) -> None:
    """Match C++ randomizeHe for Conv2d / Linear (Kaiming normal for ReLU)."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_optimizer_cpp_style(model: nn.Module) -> torch.optim.SGD:
    """C++-style: decay on Conv/BN gamma/Linear weights only; no decay on any bias."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            decay.append(m.weight)
            if m.bias is not None:
                no_decay.append(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            decay.append(m.weight)
            no_decay.append(m.bias)
        elif isinstance(m, nn.Linear):
            decay.append(m.weight)
            if m.bias is not None:
                no_decay.append(m.bias)
    return torch.optim.SGD(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=LR,
        momentum=0.0,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=1, help=f"C++ demo default is {EPOCHS_DEFAULT}")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--data-dir", type=str, default="./data", help="Root for torchvision CIFAR-10 download")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    p.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            # CIFAR-10 ToTensor() yields [0,1]; C++ loader uses raw bytes / 255 — same scale.
        ]
    )
    train_set = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=tfm)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model = ConvDemoCudaArch().to(device)
    model.apply(he_init_relu)
    opt = build_optimizer_cpp_style(model)
    criterion = nn.CrossEntropyLoss()

    print(
        f"PyTorch {torch.__version__} device={device}  batches/epoch={len(loader)}  "
        f"batch_size={args.batch_size}"
    )

    for epoch in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize()
        running_loss = 0.0
        n_batches = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            running_loss += float(loss.detach())
            n_batches += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        mean_loss = running_loss / max(n_batches, 1)
        print(
            f"epoch {epoch + 1}/{args.epochs}  "
            f"time_s={elapsed:.4f}  s/epoch={elapsed:.4f}  mean_loss={mean_loss:.4f}"
        )


if __name__ == "__main__":
    main()
