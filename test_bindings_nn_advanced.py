"""Advanced Sequential binding tests for parameter sync/introspection.

Run:
  PYTHONPATH="$PWD/python" python3 test_bindings_nn_advanced.py
"""

from __future__ import annotations

from ctorch import Matrix, Sequential


def _assert_close(a: float, b: float, tol: float = 1e-9) -> None:
    assert abs(a - b) <= tol, f"expected {a} ~= {b} (tol={tol})"


def test_copy_parameters_from_synchronizes_outputs() -> None:
    src = Sequential().add_linear(2, 2).add_relu().add_linear(2, 1)
    dst = Sequential().add_linear(2, 2).add_relu().add_linear(2, 1)

    # Source parameters (W0, b0, W1, b1)
    src.set_parameter(0, [[1.0, -2.0], [0.5, 3.0]])
    src.set_parameter(1, [[0.25], [-1.0]])
    src.set_parameter(2, [[2.0, -1.5]])
    src.set_parameter(3, [[0.75]])

    # Different destination parameters to verify sync actually changes behavior.
    dst.set_parameter(0, [[0.0, 0.0], [0.0, 0.0]])
    dst.set_parameter(1, [[0.0], [0.0]])
    dst.set_parameter(2, [[0.0, 0.0]])
    dst.set_parameter(3, [[0.0]])

    x = Matrix([[1.2], [-0.4]])
    src_out = src.forward(x).get(0, 0)
    dst_before = dst.forward(x).get(0, 0)
    assert abs(src_out - dst_before) > 1e-6

    dst.copy_parameters_from(src)
    dst_after = dst.forward(x).get(0, 0)
    _assert_close(src_out, dst_after, tol=1e-12)


def test_get_set_parameter_updates_behavior() -> None:
    model = Sequential().add_linear(1, 1)

    assert model.parameter_count == 2

    model.set_parameter(0, [[2.0]])
    model.set_parameter(1, Matrix([[0.5]]))

    assert model.get_parameter(0).to_list() == [[2.0]]
    assert model.get_parameter(1).to_list() == [[0.5]]

    y_before = model.forward([[3.0]]).get(0, 0)
    _assert_close(y_before, 6.5)

    model.set_parameter(1, [[1.5]])
    y_after = model.forward([[3.0]]).get(0, 0)
    _assert_close(y_after, 7.5)


def test_linear_layer_dims_multilayer() -> None:
    model = Sequential().add_linear(3, 4).add_relu().add_linear(4, 2).add_tanh().add_linear(2, 1)

    assert model.linear_layer_count == 3
    assert model.linear_layer_dims(0) == (3, 4)
    assert model.linear_layer_dims(1) == (4, 2)
    assert model.linear_layer_dims(2) == (2, 1)


def main() -> None:
    test_copy_parameters_from_synchronizes_outputs()
    test_get_set_parameter_updates_behavior()
    test_linear_layer_dims_multilayer()
    print("test_bindings_nn_advanced.py: PASS")


if __name__ == "__main__":
    main()
