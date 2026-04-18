"""Low-level symbolic binding tests for Expr differentiation and raw expression optimization.

Run:
  PYTHONPATH="$PWD/python" python3 test_bindings_symbolic_lowlevel.py
"""

from __future__ import annotations

from ctorch import Expr, OptimType, optimize_expr


def test_expr_diff_single_quadratic() -> None:
    x = Expr.var("x")
    f = x * x
    d = f.diff_single("x", 3.0)
    assert abs(d.evaluate({}) - 6.0) < 1e-3


def test_optimize_expr_convex_parabola_gd_and_adam() -> None:
    x = Expr.var("x")
    target = Expr.num(4.0)
    loss = (x - target) * (x - target) + Expr.num(1.0)
    initial = {"x": -3.0}

    theta_gd = optimize_expr(
        loss,
        initial,
        optim_type=OptimType.GD,
        learning_rate=0.1,
        max_iter=1200,
    )
    assert abs(theta_gd["x"] - 4.0) < 0.05

    theta_adam = optimize_expr(
        loss,
        initial,
        optim_type=OptimType.ADAM,
        learning_rate=0.2,
        max_iter=800,
    )
    assert abs(theta_adam["x"] - 4.0) < 0.05


if __name__ == "__main__":
    test_expr_diff_single_quadratic()
    test_optimize_expr_convex_parabola_gd_and_adam()
    print("ok")
