"""Logistic regression demo using the Python bindings.

Run from the repository root:

  python3 demos/logistic_regression/main.py

If the shared library is not already built, run:

  make py-bindings
"""

from __future__ import annotations

import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
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


from ctorch import DataAugmentationType, LogisticRegression, Matrix, OptimType  # noqa: E402
from demos.visualization import plot_binary_decision_boundary  # noqa: E402


@dataclass(frozen=True)
class ClassificationSample:
    x1: float
    x2: float
    label: int


@contextmanager
def silence_c_stdout() -> Iterator[None]:
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


def generate_blob_samples(
    count: int,
    center_x: float,
    center_y: float,
    spread: float,
    label: int,
    rng: random.Random,
) -> list[ClassificationSample]:
    samples: list[ClassificationSample] = []
    for _ in range(count):
        samples.append(
            ClassificationSample(
                rng.gauss(center_x, spread),
                rng.gauss(center_y, spread),
                label,
            )
        )
    return samples


def samples_to_features(samples: list[ClassificationSample]) -> Matrix:
    return Matrix([[sample.x1, sample.x2] for sample in samples])


def samples_to_labels(samples: list[ClassificationSample]) -> Matrix:
    return Matrix([[float(sample.label) for sample in samples]])


def sample_to_row(sample: ClassificationSample) -> Matrix:
    return Matrix([[sample.x1, sample.x2]])


def main() -> None:
    rng = random.Random(7)
    samples_per_class = 8
    train_per_class = 6
    spread = 0.25

    class_zero = generate_blob_samples(samples_per_class, -1.2, -1.0, spread, 0, rng)
    class_one = generate_blob_samples(samples_per_class, 1.2, 1.0, spread, 1, rng)

    samples = class_zero + class_one
    rng.shuffle(samples)

    train_samples = samples[: 2 * train_per_class]
    test_samples = samples[2 * train_per_class :]

    x_train = samples_to_features(train_samples)
    y_train = samples_to_labels(train_samples)
    x_test = samples_to_features(test_samples)
    y_test = samples_to_labels(test_samples)

    with silence_c_stdout():
        model = LogisticRegression(
            x_train,
            y_train,
            optim_type=OptimType.GD,
            learning_rate=0.05,
            max_iter=120,
            augmentation=DataAugmentationType.NO_OP,
        )

    train_accuracy = model.score(x_train, y_train)
    test_accuracy = model.score(x_test, y_test)
    plot_path = plot_binary_decision_boundary(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        title="Logistic Regression Decision Boundary",
        filename="logistic_regression_boundary.png",
    )

    print("Logistic Regression demo")
    print("Class 0 center: (-1.2, -1.0)")
    print("Class 1 center: (1.2, 1.0)")
    print(f"Training samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Train accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    print(f"Saved plot: {plot_path}")
    print()
    print("Test predictions:")
    for sample in test_samples:
        prediction = model.predict(sample_to_row(sample))
        print(
            f"  ({sample.x1:+.3f}, {sample.x2:+.3f}) "
            f"label={sample.label} prediction={prediction}"
        )


if __name__ == "__main__":
    main()
