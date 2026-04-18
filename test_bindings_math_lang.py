"""Smoke tests for matrix extras, symbolic Expr, symbolic_optimize, LP/QP.

Run:
  PYTHONPATH="$PWD/python" python3 test_bindings_math_lang.py
"""

from __future__ import annotations

from ctorch import (
    Expr,
    LinearProgramSense,
    Matrix,
    OptimType,
    SymbolicLoss,
    regression_label_row,
    solve_linear_program,
    solve_quadratic_program,
    symbolic_optimize,
)


def test_matrix_extras() -> None:
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    assert m.row(0).to_list() == [[1.0, 2.0]]
    assert Matrix.eye(2).to_list() == [[1.0, 0.0], [0.0, 1.0]]
    assert m.relu().to_list() == [[1.0, 2.0], [3.0, 4.0]]
    flat = Matrix([[1.0, 2.0, 3.0, 4.0]])
    assert abs(flat.inner_product(flat) - 30.0) < 1e-9


def test_expr_eval_and_grad() -> None:
    x = Expr.var("x")
    f = (x * x).simplify()
    assert abs(f.evaluate({"x": 3.0}) - 9.0) < 1e-9
    g = f.gradient({"x": 3.0})
    assert abs(g["x"] - 6.0) < 1e-3


def test_symbolic_regression_line() -> None:
    # y = 3 x + 1  (targets encoded for SymbolicLoss.regression_mse)
    x_train = Matrix([[0.0], [1.0], [2.0]])
    y_train = regression_label_row([1.0, 4.0, 7.0])
    loss = SymbolicLoss.regression_mse(1, l2=1e-6)
    theta = symbolic_optimize(
        loss,
        x_train,
        y_train,
        optim_type=OptimType.GD,
        learning_rate=0.15,
        max_iter=4000,
        batch_size=3,
    )
    assert abs(theta["w0"] - 3.0) < 0.08
    assert abs(theta["b"] - 1.0) < 0.08


def test_lp_qp() -> None:
    # min x  s.t. x <= 2, x >= 0  => x* = 0
    r = solve_linear_program(Matrix([[1.0]]), [2.0], [1.0], sense=LinearProgramSense.MINIMIZE)
    assert r.optimal and not r.unbounded
    assert abs(r.solution[0]) < 1e-6

    # min 0.5 x^2 on [0, 1]  => x = 0
    q = Matrix([[1.0]])
    qp = solve_quadratic_program(q, [0.0], [0.0], [1.0])
    assert qp.converged
    assert abs(qp.solution[0]) < 1e-2


if __name__ == "__main__":
    test_matrix_extras()
    test_expr_eval_and_grad()
    test_symbolic_regression_line()
    test_lp_qp()
    print("ok")
