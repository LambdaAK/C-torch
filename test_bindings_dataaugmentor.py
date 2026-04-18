"""Root-level smoke tests for DataAugmentor Python bindings.

Run:
  PYTHONPATH="$PWD/python" python3 test_bindings_dataaugmentor.py
"""

from __future__ import annotations

from math import isfinite

from ctorch import DataAugmentationType, Matrix, augment_data, random_fourier_features


def _expected_poly(data: list[list[float]], degree: int) -> list[list[float]]:
    out: list[list[float]] = []
    for row in data:
        expanded: list[float] = []
        for p in range(1, degree + 1):
            for value in row:
                expanded.append(value**p)
        out.append(expanded)
    return out


def _assert_matrix_close(actual: list[list[float]], expected: list[list[float]], tol: float = 1e-12) -> None:
    assert len(actual) == len(expected), f"row mismatch: {len(actual)} != {len(expected)}"
    for i, (a_row, e_row) in enumerate(zip(actual, expected)):
        assert len(a_row) == len(e_row), f"col mismatch in row {i}: {len(a_row)} != {len(e_row)}"
        for j, (a_value, e_value) in enumerate(zip(a_row, e_row)):
            diff = abs(a_value - e_value)
            assert diff <= tol, f"value mismatch at ({i}, {j}): {a_value} != {e_value} (diff={diff})"


def _assert_all_finite(values: list[list[float]]) -> None:
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            assert isfinite(value), f"non-finite value at ({i}, {j}): {value}"


def test_no_op_and_poly_shape_and_determinism() -> None:
    raw = [[1.0, -2.0], [0.5, 3.0]]
    x = Matrix(raw)
    degree_by_aug = {
        DataAugmentationType.NO_OP: 1,
        DataAugmentationType.POLY_2: 2,
        DataAugmentationType.POLY_3: 3,
        DataAugmentationType.POLY_4: 4,
        DataAugmentationType.POLY_5: 5,
    }

    for augmentation, degree in degree_by_aug.items():
        y1 = augment_data(x, augmentation=augmentation)
        y2 = augment_data(x, augmentation=augmentation)
        expected = _expected_poly(raw, degree)

        assert y1.shape == (len(expected), len(expected[0]))
        assert y2.shape == y1.shape
        _assert_matrix_close(y1.to_list(), expected)
        _assert_matrix_close(y2.to_list(), expected)


def test_rff_shape_and_finite_values() -> None:
    raw = [[0.0, 1.0], [1.0, -1.0], [2.0, 0.5]]
    x = Matrix(raw)
    d_features = 16
    gamma = 0.75

    y_rff = random_fourier_features(x, d_features=d_features, gamma=gamma)
    assert y_rff.shape == (len(raw), d_features)
    _assert_all_finite(y_rff.to_list())

    y_dispatch = augment_data(
        x,
        augmentation=DataAugmentationType.RFF,
        d_features=d_features,
        gamma=gamma,
    )
    assert y_dispatch.shape == (len(raw), d_features)
    _assert_all_finite(y_dispatch.to_list())


def main() -> None:
    test_no_op_and_poly_shape_and_determinism()
    test_rff_shape_and_finite_values()
    print("test_bindings_dataaugmentor.py: PASS")


if __name__ == "__main__":
    main()
