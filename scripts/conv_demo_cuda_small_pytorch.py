#!/usr/bin/env python3
"""
CIFAR-10 — small debug topology matching `src/app/conv_demo_cuda_small.cpp`.

Architecture (BN-stabilized, dropout after ReLUs, hidden Linear head):
  Conv(3→32)   + BN + ReLU + Dropout(0.5) + MaxPool(2,2)   # 32→16
  Conv(32→64)  + BN + ReLU + Dropout(0.5) + MaxPool(2,2)   # 16→8
  Conv(64→128) + BN + ReLU + Dropout(0.5) + MaxPool(2,2)   # 8→4
  Flatten (4*4*128 = 2048) → Linear(2048→128) + BN1d + ReLU + Dropout(0.5) → Linear(128→10)

Training defaults track the C++ demo: SGD Nesterov 0.9, weight_decay=1e-4 on conv/linear
weights only, batch 512, BN eps=1e-5 / momentum=0.01, warmup 5 epochs (0.01→0.1),
then ×0.5 every 20 epochs, `drop_last=True`. Keep constants below in sync when you
change the C++ file.

Eval/validation/test use torch.inference_mode() so no autograd graph is built; the train
pass still uses autograd for loss.backward() and optimizer.step().

Usage:
  pip install torch torchvision
  python scripts/conv_demo_cuda_small_pytorch.py
  python scripts/conv_demo_cuda_small_pytorch.py --epochs 2 --cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Match `conv_demo_cuda_small.cpp` (opt.m_batch_size, opt.m_l2_regularizer, LR schedule).
BATCH_SIZE = 512
EPOCHS = 40
L2 = 1e-4
BN_EPS = 1e-5
BN_MOMENTUM = 0.01
LR_WARMUP_START = 0.01
LR_BASE = 0.1
WARMUP_EPOCHS = 5
LR_DROP_EVERY = 20
TRAIN_N = 50_000
NUM_CLASSES = 10
# After three MaxPool(2,2): 32→16→8→4; final channels = 128 → flat 4*4*128 = 2048.
FLATTEN_DIM = 4 * 4 * 128
HIDDEN_DIM = 128
# Match `conv_demo_cuda_small.cpp` `k_dropout_keep = 0.5` (standard dropout on activations).
DROPOUT_P = 0.5


def learning_rate_for_epoch(epoch: int) -> float:
    """Match `conv_demo_cuda_small.cpp` step schedule (warmup uses epoch+1)."""
    if epoch < WARMUP_EPOCHS:
        t = (epoch + 1) / float(WARMUP_EPOCHS)
        return float(LR_WARMUP_START + t * (LR_BASE - LR_WARMUP_START))
    s = (epoch - WARMUP_EPOCHS) // LR_DROP_EVERY
    return float(LR_BASE * (0.5**s))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """L2 only on conv/linear weights — biases and any BN params get weight_decay=0."""
    decay, no_decay = [], []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            decay.append(m.weight)
            if m.bias is not None:
                no_decay.append(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.weight is not None:
                no_decay.append(m.weight)
            if m.bias is not None:
                no_decay.append(m.bias)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class SmallConvNet(nn.Module):
    """Mirror of the small C++ topology: 3× (Conv+BN+ReLU+Dropout+Pool) → FC+BN+ReLU+Dropout → FC."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=True),
            nn.BatchNorm2d(32, eps=BN_EPS, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_P),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.BatchNorm2d(64, eps=BN_EPS, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_P),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1, bias=True),
            nn.BatchNorm2d(128, eps=BN_EPS, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_P),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FLATTEN_DIM, HIDDEN_DIM, bias=True),
            nn.BatchNorm1d(HIDDEN_DIM, eps=BN_EPS, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_P),
            nn.Linear(HIDDEN_DIM, num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train: bool,
) -> tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    grad_ctx = nullcontext() if train else torch.inference_mode()
    with grad_ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            bs = x.size(0)
            tot_loss += loss.item() * bs
            tot_acc += accuracy(logits, y) * bs
            n += bs
    return tot_loss / n, tot_acc / n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--data-dir", type=str, default="./data", help="Root for CIFAR-10 download")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument("--workers", type=int, default=2, help="DataLoader num_workers")
    a = p.parse_args()

    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    raw_train = datasets.CIFAR10(
        root=a.data_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    x_01 = raw_train.data[:TRAIN_N].astype(np.float32) / 255.0
    mean = float(x_01.mean())
    std = float(x_01.std())
    print(
        f"CIFAR-10 small conv (matches conv_demo_cuda_small.cpp)  |  mean={mean:.6f}  std={std:.6f}"
    )

    m3 = (mean, mean, mean)
    s3 = (std, std, std)

    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(m3, s3),
        ]
    )
    transform_eval = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(m3, s3)]
    )

    train_set = datasets.CIFAR10(
        root=a.data_dir, train=True, download=True, transform=transform_train
    )
    test_set = datasets.CIFAR10(
        root=a.data_dir, train=False, download=True, transform=transform_eval
    )

    train_loader = DataLoader(
        train_set,
        batch_size=a.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=a.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(a.workers > 0),
    )
    val_loader = DataLoader(
        test_set,
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(a.workers > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(a.workers > 0),
    )

    model = SmallConvNet().to(device)
    opt = torch.optim.SGD(
        build_param_groups(model, L2),
        lr=LR_WARMUP_START,
        momentum=0.9,
        nesterov=True,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(a.epochs):
        set_optimizer_lr(opt, learning_rate_for_epoch(epoch))
        lr = opt.param_groups[0]["lr"]
        t0 = time.perf_counter()
        tr_loss, tr_acc = one_epoch(
            model, train_loader, device, opt, criterion, train=True
        )
        va_loss, va_acc = one_epoch(
            model, val_loader, device, opt, criterion, train=False
        )
        epoch_s = time.perf_counter() - t0
        print(
            f"Epoch {epoch + 1}/{a.epochs}  loss: {tr_loss:.4f}  val_loss: {va_loss:.4f}  "
            f"acc: {tr_acc:.4f}  val_acc: {va_acc:.4f}  lr: {lr:.6f}  time_s: {epoch_s:.3f}"
        )
        sys.stdout.flush()

    te_loss, te_acc = one_epoch(
        model, test_loader, device, opt, criterion, train=False
    )
    print(
        f"Test  loss: {te_loss:.4f}  acc: {te_acc:.4f}  (device: {device})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
