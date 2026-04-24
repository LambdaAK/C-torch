"""Distributed XOR-like neural-network demo for ctorch Sequential models.

Run locally:
  make py FILE=examples/python/train_xor_distributed.py

Run across two nodes or two local terminals:
  make py FILE=examples/python/train_xor_distributed.py ARGS="--distributed --rank 0 --world-size 2"
  make py FILE=examples/python/train_xor_distributed.py ARGS="--distributed --rank 1 --world-size 2"

Rank 0 should start first so it can bind the TCP master port before rank 1 connects.

Optional checkpointing:
  make py FILE=examples/python/train_xor_distributed.py ARGS="--checkpoint-prefix artifacts/py_models/xor_distributed"
  make py FILE=examples/python/train_xor_distributed.py ARGS="--distributed --rank 0 --world-size 2 --checkpoint-prefix artifacts/py_models/xor_distributed"
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ctorch import Matrix, NNOptimType, NNOptimizer, Sequential, TcpProcessGroup

DEFAULT_CHECKPOINT_PREFIX = Path("artifacts/py_models/xor_distributed")
DEFAULT_EPOCHS = 800
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_LOG_EVERY = 100
DEFAULT_HIDDEN_DIM = 8
GRID_SIZE = 10


@dataclass(frozen=True)
class Sample:
    x0: float
    x1: float
    label: float


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    log_every: int = DEFAULT_LOG_EVERY


@dataclass(frozen=True)
class DistributedConfig:
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    master_address: str = "127.0.0.1"
    master_port: int = 29500
    checkpoint_prefix: Path = DEFAULT_CHECKPOINT_PREFIX
    resume_checkpoint: bool = False


def make_dataset() -> list[Sample]:
    samples: list[Sample] = []
    step = 2.0 / float(GRID_SIZE - 1)
    for row in range(GRID_SIZE):
        x0 = -1.0 + step * float(row)
        for col in range(GRID_SIZE):
            x1 = -1.0 + step * float(col)
            label = 1.0 if ((x0 > 0.0) != (x1 > 0.0)) else 0.0
            samples.append(Sample(x0=x0, x1=x1, label=label))
    return samples


def shard_samples(samples: list[Sample], rank: int, world_size: int) -> list[Sample]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")
    if len(samples) % world_size != 0:
        raise ValueError("sample count must be divisible by world_size")
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


def feature_column(sample: Sample) -> Matrix:
    return Matrix([[sample.x0], [sample.x1]])


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def predict_probability(model: Sequential, sample: Sample) -> float:
    logits = model.forward(feature_column(sample)).get(0, 0)
    return sigmoid(logits)


def binary_cross_entropy(probability: float, label: float) -> float:
    clipped = min(max(probability, 1e-7), 1.0 - 1e-7)
    return -(label * math.log(clipped) + (1.0 - label) * math.log(1.0 - clipped))


def accuracy(model: Sequential, samples: list[Sample]) -> float:
    correct = 0
    for sample in samples:
        probability = predict_probability(model, sample)
        predicted = 1.0 if probability >= 0.5 else 0.0
        if predicted == sample.label:
            correct += 1
    return float(correct) / float(len(samples))


def train_and_save(cfg: TrainConfig, dist: DistributedConfig) -> None:
    samples = make_dataset()
    local_samples = shard_samples(samples, dist.rank, dist.world_size) if dist.enabled else samples

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
            group.load_checkpoint(str(dist.checkpoint_prefix), model, optimizer)
        else:
            group.synchronize_sequential_model(model)

    for epoch in range(cfg.epochs):
        optimizer.zero_grad()
        local_loss = 0.0

        local_correct = 0
        for sample in local_samples:
            logit = model.forward(feature_column(sample)).get(0, 0)
            probability = sigmoid(logit)
            local_loss += binary_cross_entropy(probability, sample.label)
            if (probability >= 0.5) == bool(sample.label):
                local_correct += 1
            model.backward(Matrix([[probability - sample.label]]))

        if group is not None:
            group.allreduce_sequential_gradients(model)
        optimizer.step()

        should_log = epoch == 0 or (epoch + 1) % cfg.log_every == 0 or (epoch + 1) == cfg.epochs
        total_loss = local_loss
        total_correct = float(local_correct)

        if group is not None:
            loss_matrix = Matrix([[local_loss]])
            correct_matrix = Matrix([[float(local_correct)]])
            group.allreduce_sum(loss_matrix)
            group.allreduce_sum(correct_matrix)
            total_loss = loss_matrix.get(0, 0)
            total_correct = correct_matrix.get(0, 0)

        if should_log and (not dist.enabled or dist.rank == 0):
            average_loss = total_loss / float(len(samples))
            current_accuracy = total_correct / float(len(samples))
            print(f"epoch {epoch + 1:4d} | loss {average_loss:.6f} | acc {current_accuracy:.3f}")

    dist.checkpoint_prefix.parent.mkdir(parents=True, exist_ok=True)
    if dist.enabled:
        if group is None:
            raise RuntimeError("distributed process group was not initialized")
        group.save_checkpoint(str(dist.checkpoint_prefix), model, optimizer)
    else:
        model.save(f"{dist.checkpoint_prefix}.model")
        optimizer.save_state(f"{dist.checkpoint_prefix}.optim")

    if not dist.enabled or dist.rank == 0:
        print()
        print("final accuracy:", f"{accuracy(model, samples):.3f}")
        print("saved checkpoint:", f"{dist.checkpoint_prefix}.model")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small XOR-like neural network with optional distributed data parallelism.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--checkpoint-prefix", type=Path, default=DEFAULT_CHECKPOINT_PREFIX)
    parser.add_argument("--resume-checkpoint", action="store_true")

    args = list(argv) if argv is not None else list(sys.argv[1:])
    return parser.parse_args(args)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = TrainConfig(
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        log_every=int(args.log_every),
    )
    dist = DistributedConfig(
        enabled=bool(args.distributed),
        rank=int(args.rank),
        world_size=int(args.world_size),
        master_address=str(args.master_addr),
        master_port=int(args.master_port),
        checkpoint_prefix=Path(args.checkpoint_prefix),
        resume_checkpoint=bool(args.resume_checkpoint),
    )
    train_and_save(cfg, dist)


if __name__ == "__main__":
    main()
