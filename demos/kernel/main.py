"""Kernel SVM concentric circles demo using the Python bindings.

Run from the repository root:

  python3 demos/kernel/main.py

If the shared library is not already built, run:

  make py-bindings
"""

from __future__ import annotations

import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from math import cos, pi, sin
from pathlib import Path
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

if "CTORCH_LIB_PATH" not in os.environ:
    for candidate in (
        REPO_ROOT / "build" / "libctorch_c.dylib",
        REPO_ROOT / "build" / "libctorch_c.so",
        REPO_ROOT / "build" / "ctorch_c.dll",
        REPO_ROOT / "build" / "libctorch_c.dll",
        REPO_ROOT / "build" / "Debug" / "libctorch_c.dylib",
        REPO_ROOT / "build" / "Release" / "libctorch_c.dylib",
        REPO_ROOT / "build" / "RelWithDebInfo" / "libctorch_c.dylib",
        REPO_ROOT / "build" / "MinSizeRel" / "libctorch_c.dylib",
    ):
        if candidate.exists():
            os.environ["CTORCH_LIB_PATH"] = str(candidate)
            break


from ctorch import KernelSVM, KernelType, Matrix  # noqa: E402


@dataclass(frozen=True)
class Sample:
    x: float
    y: float
    label: int


@contextmanager
def silence_c_stdout() -> Iterator[None]:
    """Temporarily redirect C stdout so the training printout stays quiet."""

    sys.stdout.flush()
    saved_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        yield
    finally:
        os.dup2(saved_fd, 1)
        os.close(devnull_fd)
        os.close(saved_fd)


def generate_circle_samples(
    count: int,
    radius: float,
    radius_noise: float,
    label: int,
    rng: random.Random,
) -> list[Sample]:
    samples: list[Sample] = []
    for _ in range(count):
        theta = rng.uniform(0.0, 2.0 * pi)
        r = radius + rng.gauss(0.0, radius_noise)
        samples.append(Sample(r * cos(theta), r * sin(theta), label))
    return samples


def samples_to_features(samples: list[Sample]) -> Matrix:
    return Matrix([[sample.x, sample.y] for sample in samples])


def samples_to_labels(samples: list[Sample]) -> Matrix:
    return Matrix([[float(sample.label) for sample in samples]])


def sample_to_row(sample: Sample) -> Matrix:
    return Matrix([[sample.x, sample.y]])


def main() -> None:
    rng = random.Random(7)
    inner_radius = 1.0
    outer_radius = 2.8
    radius_noise = 0.02
    samples_per_class = 8
    train_per_class = 6
    learning_rate = 0.01
    max_iter = 100
    c_value = 1.0
    gamma = 1.5

    inner = generate_circle_samples(samples_per_class, inner_radius, radius_noise, -1, rng)
    outer = generate_circle_samples(samples_per_class, outer_radius, radius_noise, 1, rng)

    all_samples = inner + outer
    rng.shuffle(all_samples)

    train_samples = all_samples[: 2 * train_per_class]
    test_samples = all_samples[2 * train_per_class :]

    x_train = samples_to_features(train_samples)
    y_train = samples_to_labels(train_samples)
    x_test = samples_to_features(test_samples)
    y_test = samples_to_labels(test_samples)

    with silence_c_stdout():
        model = KernelSVM(
            x_train,
            y_train,
            learning_rate=learning_rate,
            max_iter=max_iter,
            c_value=c_value,
            kernel=KernelType.RADIAL_BASIS,
            gamma=gamma,
        )

    train_accuracy = model.score(x_train, y_train)
    test_accuracy = model.score(x_test, y_test)

    print("Kernel SVM")
    print("Concentric circles demo (Python bindings)")
    print(f"Class -1: inner circle, radius {inner_radius:.3f}")
    print(f"Class +1: outer circle, radius {outer_radius:.3f}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Kernel: radial basis (gamma={gamma:.3f})")
    print(f"Train accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    print()
    print("Test predictions:")
    for sample in test_samples:
        prediction = model.predict(sample_to_row(sample))
        print(
            f"  ({sample.x:.3f}, {sample.y:.3f}) "
            f"label={sample.label} prediction={prediction}"
        )


if __name__ == "__main__":
    main()
