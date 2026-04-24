"""Distributed soft-margin kernel-SVM style demo on two concentric circles.

This uses the distributed random-Fourier approximation so the 1000-point
problem stays practical.
"""

from __future__ import annotations

import argparse
import math
from typing import Sequence

from ctorch import Matrix, RandomFourierSVM, TcpProcessGroup


def make_data(n: int = 1000) -> tuple[Matrix, Matrix]:
    points: list[list[float]] = []
    labels: list[float] = []
    half = n // 2
    for i in range(half):
        a = 2.0 * math.pi * float(i) / float(half)
        points.append([math.cos(a), math.sin(a)])
        labels.append(-1.0)
        points.append([3.0 * math.cos(a), 3.0 * math.sin(a)])
        labels.append(1.0)
    return Matrix(points), Matrix([labels])


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--master-addr", type=str, default="127.0.0.1")
    p.add_argument("--master-port", type=int, default=29500)
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args(list(argv) if argv is not None else None)

    x_train, y_train = make_data()
    group = TcpProcessGroup(args.master_addr, args.master_port, args.rank, args.world_size)
    model = RandomFourierSVM.train_distributed(
        x_train,
        y_train,
        16,
        1.0,
        0.03,
        args.epochs,
        1.0,
        group=group,
    )
    if args.rank == 0:
        print(f"accuracy: {model.score(x_train, y_train):.3f}")


if __name__ == "__main__":
    main()
