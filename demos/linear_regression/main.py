"""Linear regression demo using the Python bindings.

Run from the repository root:

  python3 demos/linear_regression/main.py

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


from ctorch import LinearRegression, Matrix  # noqa: E402
from demos.visualization import plot_regression_fit  # noqa: E402


@dataclass(frozen=True)
class RegressionSample:
    x: float
    y: float


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


def generate_samples(
    count: int,
    slope: float,
    intercept: float,
    noise_std: float,
    rng: random.Random,
) -> list[RegressionSample]:
    samples: list[RegressionSample] = []
    for _ in range(count):
        x = rng.uniform(-2.0, 2.0)
        y = slope * x + intercept + rng.gauss(0.0, noise_std)
        samples.append(RegressionSample(x, y))
    return samples


def samples_to_features(samples: list[RegressionSample]) -> Matrix:
    return Matrix([[sample.x] for sample in samples])


def samples_to_targets(samples: list[RegressionSample]) -> Matrix:
    return Matrix([[sample.y for sample in samples]])


def sample_to_row(x: float) -> Matrix:
    return Matrix([[x]])


def mean_absolute_error(model: LinearRegression, samples: list[RegressionSample]) -> float:
    total = 0.0
    for sample in samples:
        total += abs(model.predict(sample_to_row(sample.x)) - sample.y)
    return total / len(samples)


def main() -> None:
    rng = random.Random(11)
    slope = 2.5
    intercept = -0.7
    noise_std = 0.03
    train_count = 12
    test_count = 4
    learning_rate = 0.05
    max_iter = 200

    train_samples = generate_samples(train_count, slope, intercept, noise_std, rng)
    test_samples = generate_samples(test_count, slope, intercept, noise_std, rng)

    x_train = samples_to_features(train_samples)
    y_train = samples_to_targets(train_samples)

    with silence_c_stdout():
        model = LinearRegression(
            x_train,
            y_train,
            learning_rate=learning_rate,
            max_iter=max_iter,
        )

    train_mae = mean_absolute_error(model, train_samples)
    test_mae = mean_absolute_error(model, test_samples)
    plot_path = plot_regression_fit(
        model,
        x_train,
        y_train,
        samples_to_features(test_samples),
        samples_to_targets(test_samples),
        title=f"Linear Regression Fit (train MAE {train_mae:.3f}, test MAE {test_mae:.3f})",
        filename="linear_regression_fit.png",
    )

    print("Linear Regression demo")
    print(f"Target function: y = {slope:.3f}x + {intercept:.3f} + noise")
    print(f"Training samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Train MAE: {train_mae:.3f}")
    print(f"Test MAE: {test_mae:.3f}")
    print(f"Saved plot: {plot_path}")
    print()
    print("Test predictions:")
    for sample in test_samples:
        prediction = model.predict(sample_to_row(sample.x))
        print(
            f"  x={sample.x:+.3f} target={sample.y:+.3f} "
            f"prediction={prediction:+.3f} abs_error={abs(prediction - sample.y):.3f}"
        )

    print()
    print("Checkpoints:")
    for x in (-1.5, -0.25, 0.75, 2.0):
        target = slope * x + intercept
        prediction = model.predict(sample_to_row(x))
        print(f"  x={x:+.2f} target={target:+.3f} prediction={prediction:+.3f}")


if __name__ == "__main__":
    main()
