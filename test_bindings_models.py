"""Root-level smoke tests for additional model bindings.

Run:
  PYTHONPATH="$PWD/python" python3 test_bindings_models.py
"""

from ctorch import (
    GaussianNB,
    KMeans,
    KernelSVM,
    KernelType,
    PCA,
    RandomFourierSVM,
)


def _assert_unit_interval(value: float, label: str) -> None:
    assert 0.0 <= value <= 1.0, f"{label} out of range: {value}"


def test_kernel_and_rff_svm() -> None:
    x = [[0.0, 0.0], [1.0, 1.0], [1.3, 0.8], [-0.8, -0.7]]
    y = [[-1.0, 1.0, 1.0, -1.0]]

    kernel_svm = KernelSVM(
        x,
        y,
        learning_rate=0.01,
        max_iter=5,
        c_value=1.0,
        kernel=KernelType.LINEAR,
    )
    pred = kernel_svm.predict([[0.9, 0.9]])
    assert pred in (-1, 1)
    _assert_unit_interval(kernel_svm.score(x, y), "KernelSVM score")

    rff = RandomFourierSVM(
        x,
        y,
        d_features=8,
        gamma=0.5,
        learning_rate=0.0,
        max_iter=50,
        c_value=1.0,
    )
    pred_rff = rff.predict([[0.9, 0.9]])
    assert pred_rff in (-1, 1)
    _assert_unit_interval(rff.score(x, y), "RandomFourierSVM score")


def test_gaussian_nb_and_kmeans() -> None:
    x = [[0.0, 0.0], [1.0, 1.0], [1.3, 0.8], [-0.8, -0.7]]
    y01 = [[0.0, 1.0, 1.0, 0.0]]

    gnb = GaussianNB(x, y01)
    pred = gnb.predict([[0.9, 1.0]])
    assert pred in (0, 1)
    _assert_unit_interval(gnb.score(x, y01), "GaussianNB score")

    kmeans = KMeans(2, x, max_iter=20)
    assignments = kmeans.assignments
    assert len(assignments) == len(x)
    assert all(isinstance(v, int) for v in assignments)


def test_pca_projection() -> None:
    centered = [[-0.5, -0.5], [0.5, 0.5], [0.7, 0.2], [-0.7, -0.2]]
    pca = PCA(centered)
    proj = pca.compute_projection(1, max_iter=25, tol=1e-6)

    assert proj.num_rows == 2
    assert proj.num_cols == 1


def main() -> None:
    test_kernel_and_rff_svm()
    test_gaussian_nb_and_kmeans()
    test_pca_projection()
    print("test_bindings_models.py: PASS")


if __name__ == "__main__":
    main()
