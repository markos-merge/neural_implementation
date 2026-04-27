#!/usr/bin/env python3
"""
CIFAR-10 training matching `src/app/conv_demo_cuda.cpp` / cifar10vgg topology — PyTorch.

- 50k train / 10k val+test (val metrics use the official CIFAR-10 test set, like `conv_demo_cuda.cpp` with all `data_batch_*.bin`).
- Scalar mean/std from the 50k train fold; normalize after geometric aug (train) or after ToTensor (val/test).
- Augment: RandomHorizontalFlip(p=0.5), RandomAffine(degrees=15, translate=0.1) (order: flip then affine, matching
  `make_cifar10_training_augmentation()` after flip+Affine stages). Note: C++ affine uses OpenCV BORDER_REFLECT_101;
  torchvision defaults to zero fill on samples — small distribution difference vs CUDA.
- SGD(lr, momentum=0.9, nesterov=True), weight_decay=1e-4 on Conv2d/Linear weights only (not BN / biases;
  matches `opt.m_l2_regularizer` in `conv_demo_cuda.cpp`).
- BN: eps=1e-5, momentum=0.01 (PyTorch convention; matches C++ / Keras EMA naming).
- Warmup 5 epochs: lr = warmup_start + ((epoch+1)/warmup_epochs)*(base-start), then step ×0.5 every 20 epochs
  (same closed form as `conv_demo_cuda.cpp` CifarVggStep branch).
- `DataLoader(..., drop_last=True)` for **training** so each epoch is `N // batch_size` steps like `conv_demo_cuda.cpp` (`momentum_sgd_optimizer.hpp` uses integer `inputs.size() / batch_size`).

Each epoch prints loss, val_loss, acc, val_acc, lr, and wall time for that epoch (seconds).

Usage:
  pip install torch torchvision
  python scripts/conv_demo_cuda_pytorch.py
  python scripts/conv_demo_cuda_pytorch.py --epochs 2 --cpu
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH_SIZE = 128
EPOCHS = 250
L2 = 1e-4
BN_EPS = 1e-5
BN_MOMENTUM = 0.01  # PyTorch: running = (1-m)*old + m*batch — matches C++ `0.01f`
LR_WARMUP_START = 0.01
LR_BASE = 0.1
WARMUP_EPOCHS = 5
LR_DROP_EVERY = 20
TRAIN_N = 50_000
NUM_CLASSES = 10


def learning_rate_for_epoch(epoch: int) -> float:
    if epoch < WARMUP_EPOCHS:
        t = (epoch + 1) / float(WARMUP_EPOCHS)
        return float(LR_WARMUP_START + t * (LR_BASE - LR_WARMUP_START))
    s = (epoch - WARMUP_EPOCHS) // LR_DROP_EVERY
    return float(LR_BASE * (0.5**s))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
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


class Cifar10Vgg(nn.Module):
    """Same layer pattern as `conv_demo_cuda` / cifar10vgg: Conv–ReLU–BN–[Dropout] …"""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        bn_eps: float = BN_EPS,
        bn_momentum: float = BN_MOMENTUM,
    ) -> None:
        super().__init__()
        b = (bn_eps, bn_momentum)

        layers_list: list[nn.Module] = []
        c = 3

        def add_conv_block(out_c: int, d: float | None) -> None:
            nonlocal c, layers_list
            layers_list.extend(
                [nn.Conv2d(c, out_c, 3, padding=1, bias=True), nn.ReLU(inplace=True)]
            )
            c = out_c
            layers_list.append(nn.BatchNorm2d(c, eps=b[0], momentum=b[1]))
            if d is not None:
                layers_list.append(nn.Dropout(d))

        # block 1
        add_conv_block(64, 0.0)
        layers_list += [
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, eps=b[0], momentum=b[1]),
            nn.MaxPool2d(2, 2),
        ]
        # block 2
        add_conv_block(128, 0.0)
        layers_list += [
            nn.Conv2d(128, 128, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128, eps=b[0], momentum=b[1]),
            nn.MaxPool2d(2, 2),
        ]
        # block 3
        add_conv_block(256, 0.0)
        add_conv_block(256, 0.0)
        layers_list += [
            nn.Conv2d(256, 256, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, eps=b[0], momentum=b[1]),
            nn.MaxPool2d(2, 2),
        ]
        # block 4
        add_conv_block(512, 0.0)
        add_conv_block(512, 0.0)
        layers_list += [
            nn.Conv2d(512, 512, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, eps=b[0], momentum=b[1]),
            nn.MaxPool2d(2, 2),
        ]
        # block 5
        add_conv_block(512, 0.0)
        add_conv_block(512, 0.0)
        layers_list += [
            nn.Conv2d(512, 512, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, eps=b[0], momentum=b[1]),
            nn.MaxPool2d(2, 2),
        ]
        c = 512  # for classifier

        self.features = nn.Sequential(*layers_list)
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Flatten(),
            nn.Linear(c * 1 * 1, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512, eps=b[0], momentum=b[1]),
            nn.Dropout(0.0),
            nn.Linear(512, num_classes, bias=True),
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
    eps = 1e-7
    print(
        f"CIFAR-10: train {TRAIN_N}, val+test 10000 (official test)  |  mean={mean:.6f}  std={std:.6f}"
    )

    m3 = (mean, mean, mean)
    s3 = (std, std, std)

    # Train: aug on [0,1] PIL path — ToTensor() then scalar normalize on tensor (same as TF preprocessing_function after aug)
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

    full_train = datasets.CIFAR10(
        root=a.data_dir, train=True, download=True, transform=transform_train
    )
    test_set = datasets.CIFAR10(
        root=a.data_dir, train=False, download=True, transform=transform_eval
    )
    train_set = full_train
    val_set = test_set

    train_loader = DataLoader(
        train_set,
        batch_size=a.batch_size,
        shuffle=True,
        drop_last=True,  # match C++: floor(N / B) SGD steps per epoch (no partial last batch)
        num_workers=a.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(a.workers > 0),
    )
    val_loader = DataLoader(
        val_set,
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

    model = Cifar10Vgg().to(device)
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
