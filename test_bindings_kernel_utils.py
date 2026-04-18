"""Root-level smoke tests for kernel utility bindings.

Run:
  PYTHONPATH="$PWD/python" python3 test_bindings_kernel_utils.py
"""

from __future__ import annotations

import math

from ctorch import Expr, KernelSVM, KernelType, kernel_value


def _dot(x: list[float], y: list[float]) -> float:
    return sum(a * b for a, b in zip(x, y))


def _sq_l2_distance(x: list[float], y: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(x, y))


def test_kernel_value_matches_manual_formulas() -> None:
    x = [1.5, -2.0, 0.5]
    y = [0.25, 3.0, -1.0]
    gamma = 0.7

    expected_linear = _dot(x, y)
    expected_poly2 = (1.0 + expected_linear) ** 2
    expected_poly3 = (1.0 + expected_linear) ** 3
    expected_rbf = math.exp(-gamma * _sq_l2_distance(x, y))

    x_row = [x]
    y_row = [y]

    assert abs(kernel_value(x_row, y_row, KernelType.LINEAR) - expected_linear) < 1e-12
    assert abs(kernel_value(x_row, y_row, KernelType.POLYNOMIAL_2) - expected_poly2) < 1e-12
    assert abs(kernel_value(x_row, y_row, KernelType.POLYNOMIAL_3) - expected_poly3) < 1e-12
    assert abs(kernel_value(x_row, y_row, KernelType.RADIAL_BASIS, gamma=gamma) - expected_rbf) < 1e-12


def test_kernel_svm_loss_expr_is_valid_and_evaluable() -> None:
    x_train = [[0.0, 0.0], [1.0, 1.0], [1.2, 0.9], [-0.8, -0.7]]
    y_train = [[-1.0, 1.0, 1.0, -1.0]]

    model = KernelSVM(
        x_train,
        y_train,
        learning_rate=0.01,
        max_iter=3,
        c_value=1.0,
        kernel=KernelType.LINEAR,
    )

    expr = model.loss_expr()
    assert isinstance(expr, Expr)

    variables = expr.variables()
    assert variables
    assert "b" in variables
    assert any(name.startswith("alpha") for name in variables)

    expr_text = str(expr)
    assert isinstance(expr_text, str)
    assert len(expr_text) > 0
    assert any(name in expr_text for name in variables)

    bindings = {name: 0.0 for name in variables}
    value = expr.evaluate(bindings)
    assert isinstance(value, float)
    assert math.isfinite(value)


def main() -> None:
    test_kernel_value_matches_manual_formulas()
    test_kernel_svm_loss_expr_is_valid_and_evaluable()
    print("test_bindings_kernel_utils.py: PASS")


if __name__ == "__main__":
    main()
