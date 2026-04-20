"""Shared plotting helpers for demo scripts.

Figures are rendered with a headless Matplotlib backend and saved under
`artifacts/plots/` by default so they work in non-interactive environments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = REPO_ROOT / "artifacts" / "plots"
MPL_CACHE_DIR = REPO_ROOT / "artifacts" / ".mplconfig"
XDG_CACHE_DIR = REPO_ROOT / "artifacts" / ".cache"

MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from ctorch import Matrix  # noqa: E402

_CLASS_COLORS = ["#4C78A8", "#F58518"]
_DECISION_CMAP = ListedColormap(["#BFD7EA", "#FAD7B5"])


def _ensure_plot_dir() -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    return PLOT_DIR


def _save_figure(fig: matplotlib.figure.Figure, filename: str) -> Path:
    path = _ensure_plot_dir() / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _row_list(matrix: Matrix) -> list[list[float]]:
    return matrix.to_list()


def _column_values(matrix: Matrix) -> list[float]:
    rows = matrix.to_list()
    if not rows:
        return []
    if len(rows) == 1:
        return [float(value) for value in rows[0]]
    return [float(row[0]) for row in rows]


def _extract_points(features: Matrix, labels: Matrix) -> tuple[list[tuple[float, float]], list[int]]:
    points = [(row[0], row[1]) for row in _row_list(features)]
    label_values = [int(round(value)) for value in labels.to_list()[0]]
    return points, label_values


def plot_binary_decision_boundary(
    model: object,
    x_train: Matrix,
    y_train: Matrix,
    x_test: Matrix,
    y_test: Matrix,
    *,
    title: str,
    filename: str,
    grid_size: int = 160,
) -> Path:
    """Render a 2D decision boundary for binary classifiers."""

    train_points, train_labels = _extract_points(x_train, y_train)
    test_points, test_labels = _extract_points(x_test, y_test)
    labels = sorted({*train_labels, *test_labels})
    if len(labels) != 2:
        raise ValueError("binary decision boundary plotting requires exactly two labels")

    label_to_index = {label: idx for idx, label in enumerate(labels)}

    xs = [point[0] for point in train_points + test_points]
    ys = [point[1] for point in train_points + test_points]
    if not xs or not ys:
        raise ValueError("cannot plot an empty dataset")

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max((x_max - x_min) * 0.2, 0.5)
    y_pad = max((y_max - y_min) * 0.2, 0.5)

    grid_x = [x_min - x_pad + (2 * x_pad) * i / max(grid_size - 1, 1) for i in range(grid_size)]
    grid_y = [y_min - y_pad + (2 * y_pad) * i / max(grid_size - 1, 1) for i in range(grid_size)]

    decision_grid: list[list[int]] = []
    for y in grid_y:
        row: list[int] = []
        for x in grid_x:
            predicted = int(model.predict(Matrix([[x, y]])))
            row.append(label_to_index[predicted])
        decision_grid.append(row)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.contourf(grid_x, grid_y, decision_grid, levels=[-0.5, 0.5, 1.5], cmap=_DECISION_CMAP, alpha=0.35)
    ax.contour(grid_x, grid_y, decision_grid, levels=[0.5], colors="#444444", linewidths=1.0)

    for label_index, label in enumerate(labels):
        class_color = _CLASS_COLORS[label_index % len(_CLASS_COLORS)]
        train_x = [point[0] for point, point_label in zip(train_points, train_labels) if point_label == label]
        train_y = [point[1] for point, point_label in zip(train_points, train_labels) if point_label == label]
        test_x = [point[0] for point, point_label in zip(test_points, test_labels) if point_label == label]
        test_y = [point[1] for point, point_label in zip(test_points, test_labels) if point_label == label]

        ax.scatter(
            train_x,
            train_y,
            s=64,
            color=class_color,
            edgecolors="white",
            linewidths=0.8,
            label=f"train {label}",
        )
        ax.scatter(
            test_x,
            test_y,
            s=70,
            color=class_color,
            marker="x",
            linewidths=2.0,
            label=f"test {label}",
        )

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=True)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return _save_figure(fig, filename)


def plot_regression_fit(
    model: object,
    x_train: Matrix,
    y_train: Matrix,
    x_test: Matrix,
    y_test: Matrix,
    *,
    title: str,
    filename: str,
    grid_size: int = 200,
) -> Path:
    """Render a 1D regression scatter plot with the fitted model curve."""

    train_x = _column_values(x_train)
    train_y = _column_values(y_train)
    test_x = _column_values(x_test)
    test_y = _column_values(y_test)
    all_x = train_x + test_x
    if not all_x:
        raise ValueError("cannot plot an empty dataset")

    x_min, x_max = min(all_x), max(all_x)
    x_pad = max((x_max - x_min) * 0.2, 0.25)
    grid_x = [x_min - x_pad + (2 * x_pad) * i / max(grid_size - 1, 1) for i in range(grid_size)]
    fit_y = [float(model.predict(Matrix([[x]]))) for x in grid_x]

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.scatter(train_x, train_y, s=55, color="#4C78A8", edgecolors="white", linewidths=0.8, label="train")
    ax.scatter(test_x, test_y, s=65, color="#F58518", marker="s", edgecolors="white", linewidths=0.8, label="test")
    ax.plot(grid_x, fit_y, color="#54A24B", linewidth=2.5, label="model fit")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    return _save_figure(fig, filename)
