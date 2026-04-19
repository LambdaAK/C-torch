"""Typed end-to-end train/infer workflow for ctorch Sequential models.

Train:
  python3 examples/python/train_infer_sequential.py train --model-path artifacts/py_models/line.model

Infer:
  python3 examples/python/train_infer_sequential.py infer --model-path artifacts/py_models/line.model --x 1.75
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ctorch import Matrix, NNOptimType, NNOptimizer, Sequential


def true_function(x: float) -> float:
    return 2.5 * x - 0.7


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 600
    learning_rate: float = 0.05
    sample_count: int = 64
    noise_std: float = 0.02
    seed: int = 42


def build_model() -> Sequential:
    return Sequential().add_linear(1, 8).add_tanh().add_linear(8, 1)


def make_dataset(cfg: TrainConfig) -> list[tuple[float, float]]:
    rng = random.Random(cfg.seed)
    samples: list[tuple[float, float]] = []
    for _ in range(cfg.sample_count):
        x = rng.uniform(-2.0, 2.0)
        y = true_function(x) + rng.gauss(0.0, cfg.noise_std)
        samples.append((x, y))
    return samples


def train_and_save(model_path: Path, cfg: TrainConfig) -> None:
    samples = make_dataset(cfg)
    model = build_model()
    optimizer = NNOptimizer(
        model,
        optim_type=NNOptimType.ADAM,
        learning_rate=cfg.learning_rate,
        batch_size=len(samples),
    )

    for _ in range(cfg.epochs):
        optimizer.zero_grad()
        for x, y in samples:
            y_hat = model.forward(Matrix([[x]])).get(0, 0)
            # MSE(0.5 * (y_hat - y)^2) gradient wrt output.
            model.backward(Matrix([[y_hat - y]]))
        optimizer.step()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    checkpoints = [-1.5, -0.25, 0.75, 2.0]
    print("x      expected     predicted")
    print("-" * 31)
    for x in checkpoints:
        expected = true_function(x)
        predicted = model.forward([[x]]).get(0, 0)
        print(f"{x:+.2f}  {expected:+.6f}  {predicted:+.6f}")
    print(f"saved model -> {model_path}")


def load_and_predict(model_path: Path, x: float) -> float:
    model = build_model()
    model.load(str(model_path))
    return model.forward([[x]]).get(0, 0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/infer demo for ctorch Sequential.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="fit a tiny regression model and save weights")
    train.add_argument("--model-path", required=True, type=Path)
    train.add_argument("--epochs", type=int, default=600)
    train.add_argument("--learning-rate", type=float, default=0.05)
    train.add_argument("--sample-count", type=int, default=64)
    train.add_argument("--noise-std", type=float, default=0.02)
    train.add_argument("--seed", type=int, default=42)

    infer = sub.add_parser("infer", help="load model and predict one value")
    infer.add_argument("--model-path", required=True, type=Path)
    infer.add_argument("--x", required=True, type=float)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "train":
        cfg = TrainConfig(
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            sample_count=int(args.sample_count),
            noise_std=float(args.noise_std),
            seed=int(args.seed),
        )
        train_and_save(Path(args.model_path), cfg)
        return

    prediction = load_and_predict(Path(args.model_path), float(args.x))
    print(f"prediction({float(args.x):+.4f}) = {prediction:+.6f}")


if __name__ == "__main__":
    main()
