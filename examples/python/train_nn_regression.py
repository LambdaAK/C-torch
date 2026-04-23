"""Typed neural-network regression demo for ctorch Sequential models.

Train with defaults:
  make py FILE=examples/python/train_nn_regression.py

Train explicitly:
  python3 examples/python/train_nn_regression.py train --model-path artifacts/py_models/nonlinear_regression.model

Train distributedly:
  make py FILE=examples/python/train_nn_regression.py ARGS="--distributed --rank 0 --world-size 2"

Checkpoint:
  make py FILE=examples/python/train_nn_regression.py ARGS="--checkpoint-prefix artifacts/py_models/nonlinear_regression --resume-checkpoint"

Infer:
  python3 examples/python/train_nn_regression.py infer --model-path artifacts/py_models/nonlinear_regression.model --x0 0.25 --x1 -0.75
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ctorch import Matrix, NNOptimType, NNOptimizer, Sequential, TcpProcessGroup

DEFAULT_MODEL_PATH = Path("artifacts/py_models/nonlinear_regression.model")
DEFAULT_GRID_SIZE = 8
DEFAULT_HIDDEN_DIM = 16
DEFAULT_EPOCHS = 1200
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_LOG_EVERY = 200


@dataclass(frozen=True)
class Sample:
    x0: float
    x1: float
    target: float


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    grid_size: int = DEFAULT_GRID_SIZE
    log_every: int = DEFAULT_LOG_EVERY


@dataclass(frozen=True)
class DistributedConfig:
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    master_address: str = "127.0.0.1"
    master_port: int = 29500
    checkpoint_prefix: Path | None = None
    resume_checkpoint: bool = False


PROBE_POINTS: list[tuple[float, float]] = [
    (-0.90, -0.90),
    (-0.25, 0.25),
    (0.15, -0.55),
    (0.75, 0.40),
]


def target_function(x0: float, x1: float) -> float:
    return math.sin(math.pi * x0) + 0.5 * math.cos(math.pi * x1)


def make_dataset(grid_size: int) -> list[Sample]:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    samples: list[Sample] = []
    step = 2.0 / float(grid_size - 1)
    for row in range(grid_size):
        x0 = -1.0 + step * float(row)
        for col in range(grid_size):
            x1 = -1.0 + step * float(col)
            samples.append(Sample(x0=x0, x1=x1, target=target_function(x0, x1)))
    return samples


def shard_samples(samples: list[Sample], rank: int, world_size: int) -> list[Sample]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")
    if len(samples) % world_size != 0:
        raise ValueError("number of samples must be divisible by world_size for distributed training")
    return samples[rank::world_size]


def build_model() -> Sequential:
    return (
        Sequential()
        .add_linear(2, DEFAULT_HIDDEN_DIM)
        .add_tanh()
        .add_linear(DEFAULT_HIDDEN_DIM, DEFAULT_HIDDEN_DIM)
        .add_tanh()
        .add_linear(DEFAULT_HIDDEN_DIM, 1)
    )


def feature_column(sample: Sample | tuple[float, float]) -> Matrix:
    if isinstance(sample, Sample):
        x0, x1 = sample.x0, sample.x1
    else:
        x0, x1 = sample
    return Matrix([[x0], [x1]])


def predict(model: Sequential, x0: float, x1: float) -> float:
    return model.forward(feature_column((x0, x1))).get(0, 0)


def mse_gradient(predicted: float, target: float) -> float:
    return predicted - target


def train_and_save(model_path: Path, cfg: TrainConfig, dist: DistributedConfig) -> None:
    samples = make_dataset(cfg.grid_size)
    local_samples = shard_samples(samples, dist.rank, dist.world_size) if dist.enabled else samples
    checkpoint_prefix = dist.checkpoint_prefix or (model_path.parent / model_path.stem)

    model = build_model()
    optimizer = NNOptimizer(
        model,
        optim_type=NNOptimType.ADAM,
        learning_rate=cfg.learning_rate,
        batch_size=len(local_samples),
    )
    group: TcpProcessGroup | None = None

    if dist.enabled:
        group = TcpProcessGroup(
            dist.master_address,
            dist.master_port,
            dist.rank,
            dist.world_size,
        )
        if dist.resume_checkpoint:
            group.load_checkpoint(str(checkpoint_prefix), model, optimizer)
        else:
            group.synchronize_sequential_model(model)
    elif dist.resume_checkpoint:
        model.load(f"{checkpoint_prefix}.model")
        optimizer.load_state(f"{checkpoint_prefix}.optim")

    for epoch in range(cfg.epochs):
        optimizer.zero_grad()
        total_loss = 0.0

        for sample in local_samples:
            predicted = predict(model, sample.x0, sample.x1)
            error = mse_gradient(predicted, sample.target)
            total_loss += 0.5 * error * error
            model.backward(Matrix([[error]]))

        if group is not None:
            group.allreduce_sequential_gradients(model)
        optimizer.step()

        should_log = (
            epoch == 0
            or (epoch + 1) % cfg.log_every == 0
            or (epoch + 1) == cfg.epochs
        )
        if dist.enabled and group is not None:
            loss_matrix = Matrix([[total_loss]])
            group.allreduce_sum(loss_matrix)
            total_loss = loss_matrix.get(0, 0)

        if should_log and (not dist.enabled or dist.rank == 0):
            average_loss = total_loss / float(len(local_samples))
            if dist.enabled:
                average_loss = total_loss / float(len(samples))
            print(f"epoch {epoch + 1:4d} | mse {average_loss:.6f}")

    if dist.enabled:
        if group is None:
            raise RuntimeError("distributed process group was not initialized")
        checkpoint_prefix.parent.mkdir(parents=True, exist_ok=True)
        group.save_checkpoint(str(checkpoint_prefix), model, optimizer)
    else:
        checkpoint_prefix.parent.mkdir(parents=True, exist_ok=True)
        model.save(f"{checkpoint_prefix}.model")
        optimizer.save_state(f"{checkpoint_prefix}.optim")

    if not dist.enabled or dist.rank == 0:
        print()
        print("x0      x1      expected     predicted")
        print("-" * 41)
        for x0, x1 in PROBE_POINTS:
            expected = target_function(x0, x1)
            predicted = predict(model, x0, x1)
            print(f"{x0:+.2f}  {x1:+.2f}  {expected:+.6f}  {predicted:+.6f}")
        if dist.enabled:
            print(f"distributed model synced at -> {checkpoint_prefix}")
        else:
            print(f"saved model -> {checkpoint_prefix}.model")


def load_and_predict(model_path: Path, x0: float, x1: float) -> float:
    if not model_path.exists():
        raise FileNotFoundError(f"model file does not exist: {model_path}")

    model = build_model()
    model.load(str(model_path))
    return predict(model, x0, x1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = list(argv) if argv is not None else list(sys.argv[1:])
    if not args:
        args = ["train"]
    elif args[0] not in {"train", "infer", "-h", "--help"}:
        args = ["train", *args]

    parser = argparse.ArgumentParser(description="Train/infer demo for a small ctorch MLP.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="fit the nonlinear regression model and save weights")
    train.add_argument("--model-path", default=DEFAULT_MODEL_PATH, type=Path)
    train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    train.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    train.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    train.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    train.add_argument("--distributed", action="store_true")
    train.add_argument("--rank", type=int, default=0)
    train.add_argument("--world-size", type=int, default=1)
    train.add_argument("--master-addr", type=str, default="127.0.0.1")
    train.add_argument("--master-port", type=int, default=29500)
    train.add_argument("--checkpoint-prefix", type=Path)
    train.add_argument("--resume-checkpoint", action="store_true")

    infer = sub.add_parser("infer", help="load a saved model and predict one point")
    infer.add_argument("--model-path", default=DEFAULT_MODEL_PATH, type=Path)
    infer.add_argument("--x0", required=True, type=float)
    infer.add_argument("--x1", required=True, type=float)

    return parser.parse_args(args)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.command == "train":
        cfg = TrainConfig(
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            grid_size=int(args.grid_size),
            log_every=int(args.log_every),
        )
        dist = DistributedConfig(
            enabled=bool(args.distributed),
            rank=int(args.rank),
            world_size=int(args.world_size),
            master_address=str(args.master_addr),
            master_port=int(args.master_port),
            checkpoint_prefix=Path(args.checkpoint_prefix) if args.checkpoint_prefix else None,
            resume_checkpoint=bool(args.resume_checkpoint),
        )
        train_and_save(Path(args.model_path), cfg, dist)
        return

    prediction = load_and_predict(Path(args.model_path), float(args.x0), float(args.x1))
    print(f"prediction({float(args.x0):+.4f}, {float(args.x1):+.4f}) = {prediction:+.6f}")


if __name__ == "__main__":
    main()
