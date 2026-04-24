"""Very small distributed logistic-regression demo on XOR-like data.

Run locally:
  make py FILE=examples/python/train_logistic_xor_distributed.py

Run on two terminals or two nodes:
  make py FILE=examples/python/train_logistic_xor_distributed.py ARGS="--rank 0 --world-size 2"
  make py FILE=examples/python/train_logistic_xor_distributed.py ARGS="--rank 1 --world-size 2"

Rank 0 should start first so it can bind the TCP master port before rank 1 connects.

This demo adds one hand-built cross feature (`x0 * x1`) so logistic regression
can learn the XOR-like pattern with a linear classifier. It uses ADAM with a
small mini-batch to keep the fit stable.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from ctorch import LogisticRegression, Matrix, OptimType, TcpProcessGroup

DEFAULT_EPOCHS = 250
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_BATCH_SIZE = 16
DEFAULT_MASTER_ADDR = "127.0.0.1"
DEFAULT_MASTER_PORT = 29500
DEFAULT_WORLD_SIZE = 1
GRID_SIZE = 10


def make_dataset() -> tuple[Matrix, Matrix]:
    points: list[list[float]] = []
    labels: list[float] = []

    step = 2.0 / float(GRID_SIZE - 1)
    for row in range(GRID_SIZE):
        x0 = -1.0 + step * float(row)
        for col in range(GRID_SIZE):
            x1 = -1.0 + step * float(col)
            points.append([x0, x1, x0 * x1])
            labels.append(1.0 if (x0 > 0.0) != (x1 > 0.0) else 0.0)

    return Matrix(points), Matrix([labels])


def probe_points() -> list[tuple[float, float]]:
    return [
        (-0.80, -0.80),
        (-0.70, 0.70),
        (0.70, -0.70),
        (0.80, 0.80),
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a distributed logistic-regression model on XOR-like data.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of optimization iterations.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=DEFAULT_WORLD_SIZE)
    parser.add_argument("--master-addr", type=str, default=DEFAULT_MASTER_ADDR)
    parser.add_argument("--master-port", type=int, default=DEFAULT_MASTER_PORT)

    args = list(argv) if argv is not None else list(sys.argv[1:])
    return parser.parse_args(args)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.rank < 0 or args.world_size <= 0 or args.rank >= args.world_size:
        raise ValueError("rank must be in [0, world_size) and world_size must be positive")

    x_train, y_train = make_dataset()
    group = TcpProcessGroup(
        args.master_addr,
        int(args.master_port),
        int(args.rank),
        int(args.world_size),
    )

    model = LogisticRegression.train_distributed(
        x_train,
        y_train,
        optim_type=OptimType.ADAM,
        learning_rate=float(args.learning_rate),
        max_iter=int(args.epochs),
        batch_size=DEFAULT_BATCH_SIZE,
        group=group,
    )

    if args.rank == 0:
        print(f"distributed logistic regression on {x_train.num_rows} samples")
        print(f"accuracy: {model.score(x_train, y_train):.3f}")
        print("probe points:")
        for x0, x1 in probe_points():
            prediction = model.predict(Matrix([[x0, x1, x0 * x1]]))
            print(f"  ({x0:+.2f}, {x1:+.2f}) -> {int(prediction)}")


if __name__ == "__main__":
    main()
