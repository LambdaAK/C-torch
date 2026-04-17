"""Root-level smoke tests for core Python bindings.

Run:
  PYTHONPATH="$PWD/python" python3 test_bindings_core.py
"""

from ctorch import (
    DataAugmentationType,
    KNN,
    LinearRegression,
    LogisticRegression,
    Matrix,
    OptimType,
    Perceptron,
    SVM,
)


def _assert_unit_interval(value: float, label: str) -> None:
    assert 0.0 <= value <= 1.0, f"{label} out of range: {value}"


def test_matrix_and_knn() -> None:
    a = Matrix([[1.0, 2.0], [3.0, 4.0]])
    b = Matrix([[5.0, 6.0], [7.0, 8.0]])

    assert a.shape == (2, 2)
    assert (a + b).to_list() == [[6.0, 8.0], [10.0, 12.0]]
    assert (a - a).to_list() == [[0.0, 0.0], [0.0, 0.0]]
    assert (a @ b).shape == (2, 2)
    assert a.T.shape == (2, 2)

    x_tr = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    y_tr = [[0.0, 1.0, 1.0]]
    knn = KNN(1, x_tr, y_tr)
    pred = knn.predict([[0.2, 0.2]])
    assert pred in (0, 1)
    _assert_unit_interval(knn.score(x_tr, y_tr), "KNN score")


def test_linear_and_logistic_regression() -> None:
    x_reg = [[0.0], [1.0], [2.0], [3.0]]
    y_reg = [[0.0, 1.0, 2.0, 3.0]]
    lin = LinearRegression(x_reg, y_reg, learning_rate=0.01, max_iter=20)

    y_hat = lin.predict([[1.5]])
    assert isinstance(y_hat, float)
    _assert_unit_interval(lin.score(x_reg, y_reg, threshold=0.5), "LinearRegression score")

    x_bin = [[0.0, 0.0], [1.0, 1.0], [1.2, 0.8], [-0.3, -0.3]]
    y_bin = [[0.0, 1.0, 1.0, 0.0]]
    logreg = LogisticRegression(
        x_bin,
        y_bin,
        optim_type=OptimType.GD,
        learning_rate=0.05,
        max_iter=10,
        augmentation=DataAugmentationType.NO_OP,
    )

    pred = logreg.predict([[0.9, 1.0]])
    assert pred in (0, 1)
    _assert_unit_interval(logreg.score(x_bin, y_bin), "LogisticRegression score")


def test_perceptron_and_svm() -> None:
    x = [[0.0, 0.0], [1.0, 1.0], [1.2, 0.8], [-0.6, -0.6]]
    y = [[-1.0, 1.0, 1.0, -1.0]]

    per = Perceptron(x, y, epochs=20)
    per_pred = per.predict([[0.9, 0.9]])
    assert per_pred in (-1, 1)
    _assert_unit_interval(per.score(x, y), "Perceptron score")

    svm = SVM(
        x,
        y,
        learning_rate=0.0,
        max_iter=100,
        c_value=1.0,
        augmentation=DataAugmentationType.NO_OP,
    )
    svm_pred = svm.predict([[0.9, 0.9]])
    assert svm_pred in (-1, 1)
    _assert_unit_interval(svm.score(x, y), "SVM score")


def main() -> None:
    test_matrix_and_knn()
    test_linear_and_logistic_regression()
    test_perceptron_and_svm()
    print("test_bindings_core.py: PASS")


if __name__ == "__main__":
    main()
