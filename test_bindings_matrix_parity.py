"""Parity tests for Python Matrix convenience APIs.

Run:
  PYTHONPATH="$PWD/python" python3 test_bindings_matrix_parity.py
"""

from ctorch import Matrix


def test_matrix_equality_true_false() -> None:
    lhs = Matrix([[1.0, 2.0], [3.0, 4.0]])
    rhs_same = Matrix([[1.0, 2.0], [3.0, 4.0]])
    rhs_diff_value = Matrix([[1.0, 2.0], [3.0, 4.1]])
    rhs_diff_shape = Matrix([[1.0, 2.0, 3.0]])

    assert lhs == rhs_same
    assert not (lhs == rhs_diff_value)
    assert not (lhs == rhs_diff_shape)


def test_matrix_rows_matches_to_list_row_wise() -> None:
    m = Matrix([[1.0, 2.0, 3.0], [4.5, 5.5, 6.5], [7.0, 8.0, 9.0]])
    expected_rows = m.to_list()

    actual_rows = m.rows()
    assert len(actual_rows) == len(expected_rows)

    for i in range(len(expected_rows)):
        assert actual_rows[i].to_list() == [expected_rows[i]]


def main() -> None:
    test_matrix_equality_true_false()
    test_matrix_rows_matches_to_list_row_wise()
    print("test_bindings_matrix_parity.py: PASS")


if __name__ == "__main__":
    main()
