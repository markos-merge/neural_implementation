#!/usr/bin/env python3
"""
CIFAR-10 training with the same stack as `src/app/conv_demo_cuda.cpp`
(cifar10vgg topology: cifar10vgg.py) — TensorFlow / Keras.

- Train 50k / val+test 10k (full Keras CIFAR-10 train; val uses the official test set, matching `conv_demo_cuda.cpp`).
- Scalar normalization (x - mean) / (std + 1e-7) from the 50k training fold.
- Augmentation: horizontal flip, ±15° rotation, ±10% width/height shift (`ImageDataGenerator`;
  `preprocessing_function` normalizes *after* each affine transform, matching the C++ order).
- SGD Nesterov momentum=0.9, L2 5e-4 on conv/dense *kernels* (Keras: `kernel_regularizer`),
- Warmup 5 epochs: LR 0.01 → 0.1, then step decay ×0.5 every 20 epochs.
- Batch 128, 250 epochs; dropout disabled (rate 0 / C++ keep_prob 1).

Each epoch prints training loss, validation loss, accuracies, learning rate, and seconds for that epoch.

Usage:
  pip install "tensorflow>=2.14"
  python scripts/conv_demo_cuda_tensorflow.py
  python scripts/conv_demo_cuda_tensorflow.py --epochs 2 --no-gpu
"""

from __future__ import annotations

import argparse
import os
import time

# Must be set before `import tensorflow` to suppress TF logs on some installs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:  # Keras 3+ without keras.preprocessing
    try:
        from keras_preprocessing.image import ImageDataGenerator
    except ImportError as e:
        raise SystemExit(
            "ImageDataGenerator not found. For TensorFlow 2.16+ try:  pip install keras-preprocessing\n"
            f"Import error: {e}"
        ) from e


# --- Hyperparameters (match `conv_demo_cuda.cpp`) ------------------------------------------------

BATCH_SIZE = 128
EPOCHS = 250
L2 = 0.0005
BN_EPS = 1e-5
BN_MOMENTUM = 0.99  # Keras/TF: EMA; C++ `0.01f` is (1 - this) in the usual Keras sense
LR_WARMUP_START = 0.01
LR_BASE = 0.1
WARMUP_EPOCHS = 5
LR_DROP_EVERY = 20
TRAIN_N = 50_000


def build_model(
    l2: float, bn_eps: float, bn_momentum: float, num_classes: int = 10
) -> keras.Model:
    """VGG-style CIFAR net matching `cifar10vgg.py` / C++ `ConvBox` + two FC (512 → 10)."""
    l2r = regularizers.l2(l2)
    L = l2r  # alias for readability
    x_in = keras.Input(shape=(32, 32, 3))
    h = x_in

    def conv_bn_relu_drop(c, d=None):
        nonlocal h
        h = layers.Conv2D(
            c, 3, padding="same", kernel_regularizer=L, use_bias=True
        )(h)
        h = layers.ReLU()(h)
        h = layers.BatchNormalization(epsilon=bn_eps, momentum=bn_momentum)(h)
        if d is not None:
            h = layers.Dropout(d)(h)

    # block 1: 64×2, pool
    conv_bn_relu_drop(64, 0.0)
    h = layers.Conv2D(64, 3, padding="same", kernel_regularizer=L, use_bias=True)(h)
    h = layers.ReLU()(h)
    h = layers.BatchNormalization(epsilon=bn_eps, momentum=bn_momentum)(h)
    h = layers.MaxPool2D(2, 2)(h)

    # block 2: 128×2, pool
    conv_bn_relu_drop(128, 0.0)
    h = layers.Conv2D(128, 3, padding="same", kernel_regularizer=L, use_bias=True)(h)
    h = layers.ReLU()(h)
    h = layers.BatchNormalization(epsilon=bn_eps, momentum=bn_momentum)(h)
    h = layers.MaxPool2D(2, 2)(h)

    # block 3: 256×3, pool
    conv_bn_relu_drop(256, 0.0)
    conv_bn_relu_drop(256, 0.0)
    h = layers.Conv2D(256, 3, padding="same", kernel_regularizer=L, use_bias=True)(h)
    h = layers.ReLU()(h)
    h = layers.BatchNormalization(epsilon=bn_eps, momentum=bn_momentum)(h)
    h = layers.MaxPool2D(2, 2)(h)

    # block 4: 512×3, pool
    conv_bn_relu_drop(512, 0.0)
    conv_bn_relu_drop(512, 0.0)
    h = layers.Conv2D(512, 3, padding="same", kernel_regularizer=L, use_bias=True)(h)
    h = layers.ReLU()(h)
    h = layers.BatchNormalization(epsilon=bn_eps, momentum=bn_momentum)(h)
    h = layers.MaxPool2D(2, 2)(h)

    # block 5: 512×3, pool
    conv_bn_relu_drop(512, 0.0)
    conv_bn_relu_drop(512, 0.0)
    h = layers.Conv2D(512, 3, padding="same", kernel_regularizer=L, use_bias=True)(h)
    h = layers.ReLU()(h)
    h = layers.BatchNormalization(epsilon=bn_eps, momentum=bn_momentum)(h)
    h = layers.MaxPool2D(2, 2)(h)

    h = layers.Dropout(0.0)(h)
    h = layers.Flatten()(h)
    h = layers.Dense(512, kernel_regularizer=l2r)(h)
    h = layers.ReLU()(h)
    h = layers.BatchNormalization(epsilon=bn_eps, momentum=bn_momentum)(h)
    h = layers.Dropout(0.0)(h)
    h = layers.Dense(num_classes, activation="softmax", kernel_regularizer=l2r)(h)

    return keras.Model(x_in, h, name="cifar10vgg_tf")


def learning_rate_for_epoch(epoch: int) -> float:
    """SGD step size *during* epoch `epoch` (0-based), matching `conv_demo_cuda.cpp` (LR updated
    at end of each epoch, so the first epoch uses `LR_WARMUP_START`)."""
    if epoch < WARMUP_EPOCHS:
        return float(
            LR_WARMUP_START
            + (epoch / float(WARMUP_EPOCHS)) * (LR_BASE - LR_WARMUP_START)
        )
    s = (epoch - WARMUP_EPOCHS) // LR_DROP_EVERY
    return float(LR_BASE * (0.5**s))


def _set_optimizer_lr(opt, v: float) -> None:
    lr = opt.learning_rate
    if hasattr(lr, "assign"):
        lr.assign(v)
    else:
        opt.learning_rate = v


class CifarVggLrCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        _set_optimizer_lr(self.model.optimizer, learning_rate_for_epoch(0))

    def on_epoch_begin(self, epoch, logs=None):
        _set_optimizer_lr(self.model.optimizer, learning_rate_for_epoch(epoch))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs (default: 250)")
    p.add_argument(
        "--no-gpu",
        action="store_true",
        help="Hide visible GPUs and train on CPU (for debugging / comparison)",
    )
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    a = p.parse_args()

    if a.no_gpu:
        tf.config.set_visible_devices([], "GPU")

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    x_tr_01 = x_train[:TRAIN_N].astype("float32") / 255.0
    y_tr = y_train[:TRAIN_N]

    mean = float(np.mean(x_tr_01))
    std = float(np.std(x_tr_01))
    eps = 1e-7
    x_test_n = (x_test.astype("float32") / 255.0 - mean) / (std + eps)
    x_val = x_test_n
    y_val = y_test

    print(
        f"CIFAR-10: train {x_tr_01.shape[0]}, val+test (official test) {x_test_n.shape[0]}"
    )
    print(f"Scalar normalization from train fold: mean={mean:.6f}  std={std:.6f}")

    model = build_model(
        l2=L2, bn_eps=BN_EPS, bn_momentum=BN_MOMENTUM, num_classes=10
    )

    opt = keras.optimizers.SGD(
        learning_rate=LR_WARMUP_START, momentum=0.9, nesterov=True
    )
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    def normalize_after_aug(image):
        return (image.astype("float32") - mean) / (std + eps)

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="reflect",
        preprocessing_function=normalize_after_aug,
    )
    train_gen = datagen.flow(
        x_tr_01, y_tr, batch_size=a.batch_size, shuffle=True, seed=42
    )
    steps_per_epoch = x_tr_01.shape[0] // a.batch_size
    if steps_per_epoch < 1:
        raise SystemExit("batch_size is larger than training set size")

    lr_callback = CifarVggLrCallback()

    t_epoch0: list[float] = [0.0]

    def on_epoch_begin_time(_epoch, _logs):
        t_epoch0[0] = time.perf_counter()

    def on_epoch_end_print(epoch, logs):
        epoch_s = time.perf_counter() - t_epoch0[0]
        print(
            f"Epoch {epoch + 1}/{a.epochs}  loss: {logs.get('loss', 0.0):.4f}  val_loss: {logs.get('val_loss', 0.0):.4f}  "
            f"acc: {logs.get('accuracy', 0.0):.4f}  val_acc: {logs.get('val_accuracy', 0.0):.4f}  "
            f"lr: {learning_rate_for_epoch(epoch):.6f}  time_s: {epoch_s:.3f}"
        )

    time_start_cb = tf.keras.callbacks.LambdaCallback(
        on_epoch_begin=on_epoch_begin_time
    )
    print_cb = tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end_print)

    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=a.epochs,
        validation_data=(x_val, y_val),
        callbacks=[lr_callback, time_start_cb, print_cb],
        verbose=0,
    )

    res = model.evaluate(x_test_n, y_test, verbose=0, return_dict=True)
    print("Test set:", res)


if __name__ == "__main__":
    main()