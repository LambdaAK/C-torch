from __future__ import annotations

import ctypes as ct
import os
from enum import IntEnum
from pathlib import Path
from typing import Sequence

__all__ = [
    "Matrix",
    "KNN",
    "LinearRegression",
    "LogisticRegression",
    "Perceptron",
    "SVM",
    "KernelSVM",
    "RandomFourierSVM",
    "GaussianNB",
    "KMeans",
    "PCA",
    "OptimType",
    "DataAugmentationType",
    "KernelType",
]


class OptimType(IntEnum):
    GD = 0
    SGD = 1
    ADAGRAD = 2
    RMSPROP = 3
    ADAM = 4
    ADAMW = 5


class DataAugmentationType(IntEnum):
    NO_OP = 0
    POLY_2 = 1
    POLY_3 = 2
    POLY_4 = 3
    POLY_5 = 4
    RFF = 5


class KernelType(IntEnum):
    LINEAR = 0
    POLYNOMIAL_2 = 1
    POLYNOMIAL_3 = 2
    RADIAL_BASIS = 3


def _candidate_library_paths() -> list[Path]:
    here = Path(__file__).resolve()
    package_dir = here.parent
    repo_root = here.parents[2]

    env_path = os.getenv("CTORCH_LIB_PATH")
    paths: list[Path] = []
    if env_path:
        paths.append(Path(env_path))

    names = [
        "libctorch_c.dylib",
        "libctorch_c.so",
        "ctorch_c.dll",
        "libctorch_c.dll",
    ]
    for name in names:
        paths.append(package_dir / name)

    build_dir = repo_root / "build"
    for name in names:
        paths.append(build_dir / name)
    for config in ("Debug", "Release", "RelWithDebInfo", "MinSizeRel"):
        for name in names:
            paths.append(build_dir / config / name)

    return paths


def _load_library() -> ct.CDLL:
    for path in _candidate_library_paths():
        if path.exists():
            return ct.CDLL(str(path))
    candidates = "\n".join(str(p) for p in _candidate_library_paths())
    raise RuntimeError(
        "Could not locate ctorch binding library.\n"
        "Build with CTORCH_BUILD_PYTHON_BINDINGS=ON or set CTORCH_LIB_PATH.\n"
        f"Tried:\n{candidates}"
    )


_lib = _load_library()

_lib.ctorch_last_error.argtypes = []
_lib.ctorch_last_error.restype = ct.c_char_p
_lib.ctorch_clear_error.argtypes = []
_lib.ctorch_clear_error.restype = None

_lib.ctorch_matrix_create.argtypes = [ct.c_size_t, ct.c_size_t]
_lib.ctorch_matrix_create.restype = ct.c_void_p
_lib.ctorch_matrix_from_array.argtypes = [
    ct.c_size_t,
    ct.c_size_t,
    ct.POINTER(ct.c_double),
    ct.c_size_t,
]
_lib.ctorch_matrix_from_array.restype = ct.c_void_p
_lib.ctorch_matrix_destroy.argtypes = [ct.c_void_p]
_lib.ctorch_matrix_destroy.restype = None

_lib.ctorch_matrix_rows.argtypes = [ct.c_void_p]
_lib.ctorch_matrix_rows.restype = ct.c_size_t
_lib.ctorch_matrix_cols.argtypes = [ct.c_void_p]
_lib.ctorch_matrix_cols.restype = ct.c_size_t

_lib.ctorch_matrix_get.argtypes = [
    ct.c_void_p,
    ct.c_size_t,
    ct.c_size_t,
    ct.POINTER(ct.c_double),
]
_lib.ctorch_matrix_get.restype = ct.c_bool
_lib.ctorch_matrix_set.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_double]
_lib.ctorch_matrix_set.restype = ct.c_bool

_lib.ctorch_matrix_to_array.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double), ct.c_size_t]
_lib.ctorch_matrix_to_array.restype = ct.c_bool

_lib.ctorch_matrix_add.argtypes = [ct.c_void_p, ct.c_void_p]
_lib.ctorch_matrix_add.restype = ct.c_void_p
_lib.ctorch_matrix_sub.argtypes = [ct.c_void_p, ct.c_void_p]
_lib.ctorch_matrix_sub.restype = ct.c_void_p
_lib.ctorch_matrix_mul_scalar.argtypes = [ct.c_void_p, ct.c_double]
_lib.ctorch_matrix_mul_scalar.restype = ct.c_void_p
_lib.ctorch_matrix_matmul.argtypes = [ct.c_void_p, ct.c_void_p]
_lib.ctorch_matrix_matmul.restype = ct.c_void_p
_lib.ctorch_matrix_transpose.argtypes = [ct.c_void_p]
_lib.ctorch_matrix_transpose.restype = ct.c_void_p

_lib.ctorch_knn_create.argtypes = [ct.c_size_t, ct.c_void_p, ct.c_void_p]
_lib.ctorch_knn_create.restype = ct.c_void_p
_lib.ctorch_knn_destroy.argtypes = [ct.c_void_p]
_lib.ctorch_knn_destroy.restype = None
_lib.ctorch_knn_predict.argtypes = [ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_int)]
_lib.ctorch_knn_predict.restype = ct.c_bool
_lib.ctorch_knn_score.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_double)]
_lib.ctorch_knn_score.restype = ct.c_bool
_lib.ctorch_knn_get_k.argtypes = [ct.c_void_p, ct.POINTER(ct.c_size_t)]
_lib.ctorch_knn_get_k.restype = ct.c_bool
_lib.ctorch_knn_set_k.argtypes = [ct.c_void_p, ct.c_size_t]
_lib.ctorch_knn_set_k.restype = ct.c_bool

_lib.ctorch_linear_regression_create.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_double,
    ct.c_int,
]
_lib.ctorch_linear_regression_create.restype = ct.c_void_p
_lib.ctorch_linear_regression_destroy.argtypes = [ct.c_void_p]
_lib.ctorch_linear_regression_destroy.restype = None
_lib.ctorch_linear_regression_predict.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_double),
]
_lib.ctorch_linear_regression_predict.restype = ct.c_bool
_lib.ctorch_linear_regression_score.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_double,
    ct.POINTER(ct.c_double),
]
_lib.ctorch_linear_regression_score.restype = ct.c_bool

_lib.ctorch_logistic_regression_create.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_double,
    ct.c_int,
    ct.c_int,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_int,
]
_lib.ctorch_logistic_regression_create.restype = ct.c_void_p
_lib.ctorch_logistic_regression_destroy.argtypes = [ct.c_void_p]
_lib.ctorch_logistic_regression_destroy.restype = None
_lib.ctorch_logistic_regression_predict.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int),
]
_lib.ctorch_logistic_regression_predict.restype = ct.c_bool
_lib.ctorch_logistic_regression_score.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_double),
]
_lib.ctorch_logistic_regression_score.restype = ct.c_bool

_lib.ctorch_perceptron_create.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
]
_lib.ctorch_perceptron_create.restype = ct.c_void_p
_lib.ctorch_perceptron_destroy.argtypes = [ct.c_void_p]
_lib.ctorch_perceptron_destroy.restype = None
_lib.ctorch_perceptron_predict.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int),
]
_lib.ctorch_perceptron_predict.restype = ct.c_bool
_lib.ctorch_perceptron_score.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_double),
]
_lib.ctorch_perceptron_score.restype = ct.c_bool

_lib.ctorch_svm_create.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_double,
    ct.c_int,
    ct.c_double,
    ct.c_int,
]
_lib.ctorch_svm_create.restype = ct.c_void_p
_lib.ctorch_svm_destroy.argtypes = [ct.c_void_p]
_lib.ctorch_svm_destroy.restype = None
_lib.ctorch_svm_predict.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int),
]
_lib.ctorch_svm_predict.restype = ct.c_bool
_lib.ctorch_svm_score.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_double),
]
_lib.ctorch_svm_score.restype = ct.c_bool

_lib.ctorch_kernel_svm_create.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_double,
    ct.c_int,
    ct.c_double,
    ct.c_int,
    ct.c_double,
]
_lib.ctorch_kernel_svm_create.restype = ct.c_void_p
_lib.ctorch_kernel_svm_destroy.argtypes = [ct.c_void_p]
_lib.ctorch_kernel_svm_destroy.restype = None
_lib.ctorch_kernel_svm_predict.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int),
]
_lib.ctorch_kernel_svm_predict.restype = ct.c_bool
_lib.ctorch_kernel_svm_score.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_double),
]
_lib.ctorch_kernel_svm_score.restype = ct.c_bool

_lib.ctorch_random_fourier_svm_create.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_double,
    ct.c_double,
    ct.c_int,
    ct.c_double,
]
_lib.ctorch_random_fourier_svm_create.restype = ct.c_void_p
_lib.ctorch_random_fourier_svm_destroy.argtypes = [ct.c_void_p]
_lib.ctorch_random_fourier_svm_destroy.restype = None
_lib.ctorch_random_fourier_svm_predict.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int),
]
_lib.ctorch_random_fourier_svm_predict.restype = ct.c_bool
_lib.ctorch_random_fourier_svm_score.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_double),
]
_lib.ctorch_random_fourier_svm_score.restype = ct.c_bool

_lib.ctorch_gaussian_nb_create.argtypes = [ct.c_void_p, ct.c_void_p]
_lib.ctorch_gaussian_nb_create.restype = ct.c_void_p
_lib.ctorch_gaussian_nb_destroy.argtypes = [ct.c_void_p]
_lib.ctorch_gaussian_nb_destroy.restype = None
_lib.ctorch_gaussian_nb_predict.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int),
]
_lib.ctorch_gaussian_nb_predict.restype = ct.c_bool
_lib.ctorch_gaussian_nb_score.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_double),
]
_lib.ctorch_gaussian_nb_score.restype = ct.c_bool

_lib.ctorch_kmeans_create.argtypes = [ct.c_int, ct.c_void_p, ct.c_int]
_lib.ctorch_kmeans_create.restype = ct.c_void_p
_lib.ctorch_kmeans_destroy.argtypes = [ct.c_void_p]
_lib.ctorch_kmeans_destroy.restype = None
_lib.ctorch_kmeans_assignment_count.argtypes = [ct.c_void_p, ct.POINTER(ct.c_size_t)]
_lib.ctorch_kmeans_assignment_count.restype = ct.c_bool
_lib.ctorch_kmeans_get_assignments.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int),
    ct.c_size_t,
]
_lib.ctorch_kmeans_get_assignments.restype = ct.c_bool

_lib.ctorch_pca_create.argtypes = [ct.c_void_p]
_lib.ctorch_pca_create.restype = ct.c_void_p
_lib.ctorch_pca_destroy.argtypes = [ct.c_void_p]
_lib.ctorch_pca_destroy.restype = None
_lib.ctorch_pca_compute_projection.argtypes = [
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
    ct.c_double,
]
_lib.ctorch_pca_compute_projection.restype = ct.c_void_p

def _last_error() -> str:
    raw = _lib.ctorch_last_error()
    if raw is None:
        return "unknown binding error"
    text = raw.decode("utf-8", errors="replace").strip()
    return text or "unknown binding error"


def _raise_last_error(prefix: str) -> RuntimeError:
    return RuntimeError(f"{prefix}: {_last_error()}")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float))


def _coerce_enum(value: int | IntEnum, enum_type: type[IntEnum], field_name: str) -> IntEnum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(int(value))
    except Exception as exc:  # pragma: no cover - defensive typing gate
        raise ValueError(f"invalid {field_name}: {value}") from exc


def _coerce_2d(data: Sequence[object]) -> tuple[int, int, list[float]]:
    if hasattr(data, "tolist"):
        data = data.tolist()

    if not isinstance(data, Sequence):
        raise TypeError("data must be a 2D sequence")

    rows = list(data)
    if not rows:
        return 0, 0, []

    if _is_number(rows[0]):
        rows = [rows]

    matrix_rows: list[list[float]] = []
    for row in rows:
        if not isinstance(row, Sequence):
            raise TypeError("every row must be a sequence")
        converted = [float(v) for v in row]
        matrix_rows.append(converted)

    n_rows = len(matrix_rows)
    n_cols = len(matrix_rows[0]) if matrix_rows else 0

    for row in matrix_rows:
        if len(row) != n_cols:
            raise ValueError("all rows must have the same length")

    flat: list[float] = []
    for row in matrix_rows:
        flat.extend(row)

    return n_rows, n_cols, flat


def _coerce_matrix(value: Matrix | Sequence[object]) -> Matrix:
    if isinstance(value, Matrix):
        return value
    return Matrix(data=value)


def _coerce_row_vector(value: Matrix | Sequence[object]) -> Matrix:
    matrix = _coerce_matrix(value)
    if matrix.num_rows != 1:
        raise ValueError("expected a row vector with shape (1, n)")
    return matrix


class Matrix:
    __slots__ = ("_ptr",)

    def __init__(
        self,
        rows: int | Sequence[object] | None = None,
        cols: int | None = None,
        data: Sequence[object] | None = None,
        _ptr: int | None = None,
    ) -> None:
        if _ptr is not None:
            self._ptr = ct.c_void_p(_ptr)
            return

        if data is None and cols is None and rows is not None and not _is_number(rows):
            data = rows
            rows = None

        if data is not None:
            if rows is not None or cols is not None:
                raise ValueError("do not combine data with rows/cols")
            n_rows, n_cols, flat = _coerce_2d(data)
            if flat:
                arr = (ct.c_double * len(flat))(*flat)
                ptr = _lib.ctorch_matrix_from_array(n_rows, n_cols, arr, len(flat))
            else:
                ptr = _lib.ctorch_matrix_from_array(n_rows, n_cols, None, 0)
            if not ptr:
                raise _raise_last_error("Matrix(data=...) failed")
            self._ptr = ct.c_void_p(ptr)
            return

        if rows is None or cols is None:
            raise ValueError("provide Matrix(data) or Matrix(rows, cols)")

        if not _is_number(rows) or not _is_number(cols):
            raise TypeError("rows and cols must be numeric")

        n_rows = int(rows)
        n_cols = int(cols)
        if n_rows < 0 or n_cols < 0:
            raise ValueError("rows and cols must be non-negative")

        ptr = _lib.ctorch_matrix_create(n_rows, n_cols)
        if not ptr:
            raise _raise_last_error("Matrix(rows, cols) failed")
        self._ptr = ct.c_void_p(ptr)

    @classmethod
    def from_list(cls, data: Sequence[object]) -> Matrix:
        return cls(data=data)

    @property
    def num_rows(self) -> int:
        return int(_lib.ctorch_matrix_rows(self._ptr))

    @property
    def num_cols(self) -> int:
        return int(_lib.ctorch_matrix_cols(self._ptr))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_rows, self.num_cols)

    def get(self, row: int, col: int) -> float:
        out = ct.c_double()
        ok = _lib.ctorch_matrix_get(self._ptr, int(row), int(col), ct.byref(out))
        if not ok:
            raise _raise_last_error("Matrix.get failed")
        return float(out.value)

    def set(self, row: int, col: int, value: float) -> None:
        ok = _lib.ctorch_matrix_set(self._ptr, int(row), int(col), float(value))
        if not ok:
            raise _raise_last_error("Matrix.set failed")

    def transpose(self) -> Matrix:
        ptr = _lib.ctorch_matrix_transpose(self._ptr)
        if not ptr:
            raise _raise_last_error("Matrix.transpose failed")
        return Matrix(_ptr=ptr)

    T = property(transpose)

    def to_list(self) -> list[list[float]]:
        rows, cols = self.shape
        count = rows * cols
        if count == 0:
            ok = _lib.ctorch_matrix_to_array(self._ptr, None, 0)
            if not ok:
                raise _raise_last_error("Matrix.to_list failed")
            return []

        arr = (ct.c_double * count)()
        ok = _lib.ctorch_matrix_to_array(self._ptr, arr, count)
        if not ok:
            raise _raise_last_error("Matrix.to_list failed")

        out: list[list[float]] = []
        idx = 0
        for _ in range(rows):
            row_values: list[float] = []
            for _ in range(cols):
                row_values.append(float(arr[idx]))
                idx += 1
            out.append(row_values)
        return out

    def __getitem__(self, key: tuple[int, int]) -> float:
        row, col = key
        return self.get(row, col)

    def __setitem__(self, key: tuple[int, int], value: float) -> None:
        row, col = key
        self.set(row, col, value)

    def __add__(self, other: Matrix | Sequence[object]) -> Matrix:
        rhs = _coerce_matrix(other)
        ptr = _lib.ctorch_matrix_add(self._ptr, rhs._ptr)
        if not ptr:
            raise _raise_last_error("Matrix.__add__ failed")
        return Matrix(_ptr=ptr)

    def __sub__(self, other: Matrix | Sequence[object]) -> Matrix:
        rhs = _coerce_matrix(other)
        ptr = _lib.ctorch_matrix_sub(self._ptr, rhs._ptr)
        if not ptr:
            raise _raise_last_error("Matrix.__sub__ failed")
        return Matrix(_ptr=ptr)

    def __matmul__(self, other: Matrix | Sequence[object]) -> Matrix:
        rhs = _coerce_matrix(other)
        ptr = _lib.ctorch_matrix_matmul(self._ptr, rhs._ptr)
        if not ptr:
            raise _raise_last_error("Matrix.__matmul__ failed")
        return Matrix(_ptr=ptr)

    def __mul__(self, scalar: float) -> Matrix:
        if not _is_number(scalar):
            return NotImplemented
        ptr = _lib.ctorch_matrix_mul_scalar(self._ptr, float(scalar))
        if not ptr:
            raise _raise_last_error("Matrix.__mul__ failed")
        return Matrix(_ptr=ptr)

    def __rmul__(self, scalar: float) -> Matrix:
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        return f"Matrix(shape={self.shape}, data={self.to_list()})"

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _lib.ctorch_matrix_destroy(ptr)
            self._ptr = ct.c_void_p()


class KNN:
    __slots__ = ("_ptr",)

    def __init__(
        self,
        k: int,
        x_train: Matrix | Sequence[object],
        y_train: Matrix | Sequence[object],
    ) -> None:
        x_mat = _coerce_matrix(x_train)
        y_mat = _coerce_row_vector(y_train)
        ptr = _lib.ctorch_knn_create(int(k), x_mat._ptr, y_mat._ptr)
        if not ptr:
            raise _raise_last_error("KNN(...) failed")
        self._ptr = ct.c_void_p(ptr)

    @property
    def k(self) -> int:
        out = ct.c_size_t()
        ok = _lib.ctorch_knn_get_k(self._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("KNN.k getter failed")
        return int(out.value)

    @k.setter
    def k(self, new_k: int) -> None:
        ok = _lib.ctorch_knn_set_k(self._ptr, int(new_k))
        if not ok:
            raise _raise_last_error("KNN.k setter failed")

    def predict(self, sample: Matrix | Sequence[object]) -> int:
        sample_mat = _coerce_row_vector(sample)
        out = ct.c_int()
        ok = _lib.ctorch_knn_predict(self._ptr, sample_mat._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("KNN.predict failed")
        return int(out.value)

    def score(
        self,
        x_test: Matrix | Sequence[object],
        y_test: Matrix | Sequence[object],
    ) -> float:
        x_mat = _coerce_matrix(x_test)
        y_mat = _coerce_row_vector(y_test)
        out = ct.c_double()
        ok = _lib.ctorch_knn_score(self._ptr, x_mat._ptr, y_mat._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("KNN.score failed")
        return float(out.value)

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _lib.ctorch_knn_destroy(ptr)
            self._ptr = ct.c_void_p()


class LinearRegression:
    __slots__ = ("_ptr",)

    def __init__(
        self,
        x_train: Matrix | Sequence[object],
        y_train: Matrix | Sequence[object],
        learning_rate: float,
        max_iter: int,
    ) -> None:
        x_mat = _coerce_matrix(x_train)
        y_mat = _coerce_row_vector(y_train)
        ptr = _lib.ctorch_linear_regression_create(
            x_mat._ptr,
            y_mat._ptr,
            float(learning_rate),
            int(max_iter),
        )
        if not ptr:
            raise _raise_last_error("LinearRegression(...) failed")
        self._ptr = ct.c_void_p(ptr)

    def predict(self, sample: Matrix | Sequence[object]) -> float:
        sample_mat = _coerce_row_vector(sample)
        out = ct.c_double()
        ok = _lib.ctorch_linear_regression_predict(self._ptr, sample_mat._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("LinearRegression.predict failed")
        return float(out.value)

    def score(
        self,
        x_test: Matrix | Sequence[object],
        y_test: Matrix | Sequence[object],
        threshold: float = 0.5,
    ) -> float:
        x_mat = _coerce_matrix(x_test)
        y_mat = _coerce_row_vector(y_test)
        out = ct.c_double()
        ok = _lib.ctorch_linear_regression_score(
            self._ptr,
            x_mat._ptr,
            y_mat._ptr,
            float(threshold),
            ct.byref(out),
        )
        if not ok:
            raise _raise_last_error("LinearRegression.score failed")
        return float(out.value)

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _lib.ctorch_linear_regression_destroy(ptr)
            self._ptr = ct.c_void_p()


class LogisticRegression:
    __slots__ = ("_ptr",)

    def __init__(
        self,
        x_train: Matrix | Sequence[object],
        y_train: Matrix | Sequence[object],
        optim_type: OptimType | int = OptimType.GD,
        learning_rate: float = 0.001,
        max_iter: int = 1000,
        batch_size: int = 1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        rho: float = 0.99,
        weight_decay: float = 0.0,
        augmentation: DataAugmentationType | int = DataAugmentationType.NO_OP,
    ) -> None:
        x_mat = _coerce_matrix(x_train)
        y_mat = _coerce_row_vector(y_train)
        optim = _coerce_enum(optim_type, OptimType, "optim_type")
        augment = _coerce_enum(augmentation, DataAugmentationType, "augmentation")

        ptr = _lib.ctorch_logistic_regression_create(
            x_mat._ptr,
            y_mat._ptr,
            int(optim),
            float(learning_rate),
            int(max_iter),
            int(batch_size),
            float(beta1),
            float(beta2),
            float(epsilon),
            float(rho),
            float(weight_decay),
            int(augment),
        )
        if not ptr:
            raise _raise_last_error("LogisticRegression(...) failed")
        self._ptr = ct.c_void_p(ptr)

    def predict(self, sample: Matrix | Sequence[object]) -> int:
        sample_mat = _coerce_row_vector(sample)
        out = ct.c_int()
        ok = _lib.ctorch_logistic_regression_predict(self._ptr, sample_mat._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("LogisticRegression.predict failed")
        return int(out.value)

    def score(
        self,
        x_test: Matrix | Sequence[object],
        y_test: Matrix | Sequence[object],
    ) -> float:
        x_mat = _coerce_matrix(x_test)
        y_mat = _coerce_row_vector(y_test)
        out = ct.c_double()
        ok = _lib.ctorch_logistic_regression_score(
            self._ptr,
            x_mat._ptr,
            y_mat._ptr,
            ct.byref(out),
        )
        if not ok:
            raise _raise_last_error("LogisticRegression.score failed")
        return float(out.value)

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _lib.ctorch_logistic_regression_destroy(ptr)
            self._ptr = ct.c_void_p()


class Perceptron:
    __slots__ = ("_ptr",)

    def __init__(
        self,
        x_train: Matrix | Sequence[object],
        y_train: Matrix | Sequence[object],
        epochs: int = 300,
    ) -> None:
        x_mat = _coerce_matrix(x_train)
        y_mat = _coerce_row_vector(y_train)
        ptr = _lib.ctorch_perceptron_create(x_mat._ptr, y_mat._ptr, int(epochs))
        if not ptr:
            raise _raise_last_error("Perceptron(...) failed")
        self._ptr = ct.c_void_p(ptr)

    def predict(self, sample: Matrix | Sequence[object]) -> int:
        sample_mat = _coerce_row_vector(sample)
        out = ct.c_int()
        ok = _lib.ctorch_perceptron_predict(self._ptr, sample_mat._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("Perceptron.predict failed")
        return int(out.value)

    def score(
        self,
        x_test: Matrix | Sequence[object],
        y_test: Matrix | Sequence[object],
    ) -> float:
        x_mat = _coerce_matrix(x_test)
        y_mat = _coerce_row_vector(y_test)
        out = ct.c_double()
        ok = _lib.ctorch_perceptron_score(self._ptr, x_mat._ptr, y_mat._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("Perceptron.score failed")
        return float(out.value)

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _lib.ctorch_perceptron_destroy(ptr)
            self._ptr = ct.c_void_p()


class SVM:
    __slots__ = ("_ptr",)

    def __init__(
        self,
        x_train: Matrix | Sequence[object],
        y_train: Matrix | Sequence[object],
        learning_rate: float,
        max_iter: int,
        c_value: float,
        augmentation: DataAugmentationType | int = DataAugmentationType.NO_OP,
    ) -> None:
        x_mat = _coerce_matrix(x_train)
        y_mat = _coerce_row_vector(y_train)
        augment = _coerce_enum(augmentation, DataAugmentationType, "augmentation")
        ptr = _lib.ctorch_svm_create(
            x_mat._ptr,
            y_mat._ptr,
            float(learning_rate),
            int(max_iter),
            float(c_value),
            int(augment),
        )
        if not ptr:
            raise _raise_last_error("SVM(...) failed")
        self._ptr = ct.c_void_p(ptr)

    def predict(self, sample: Matrix | Sequence[object]) -> int:
        sample_mat = _coerce_row_vector(sample)
        out = ct.c_int()
        ok = _lib.ctorch_svm_predict(self._ptr, sample_mat._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("SVM.predict failed")
        return int(out.value)

    def score(
        self,
        x_test: Matrix | Sequence[object],
        y_test: Matrix | Sequence[object],
    ) -> float:
        x_mat = _coerce_matrix(x_test)
        y_mat = _coerce_row_vector(y_test)
        out = ct.c_double()
        ok = _lib.ctorch_svm_score(self._ptr, x_mat._ptr, y_mat._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("SVM.score failed")
        return float(out.value)

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _lib.ctorch_svm_destroy(ptr)
            self._ptr = ct.c_void_p()


class KernelSVM:
    __slots__ = ("_ptr",)

    def __init__(
        self,
        x_train: Matrix | Sequence[object],
        y_train: Matrix | Sequence[object],
        learning_rate: float,
        max_iter: int,
        c_value: float,
        kernel: KernelType | int = KernelType.LINEAR,
        gamma: float = 1.0,
    ) -> None:
        x_mat = _coerce_matrix(x_train)
        y_mat = _coerce_row_vector(y_train)
        kernel_type = _coerce_enum(kernel, KernelType, "kernel")
        ptr = _lib.ctorch_kernel_svm_create(
            x_mat._ptr,
            y_mat._ptr,
            float(learning_rate),
            int(max_iter),
            float(c_value),
            int(kernel_type),
            float(gamma),
        )
        if not ptr:
            raise _raise_last_error("KernelSVM(...) failed")
        self._ptr = ct.c_void_p(ptr)

    def predict(self, sample: Matrix | Sequence[object]) -> int:
        sample_mat = _coerce_row_vector(sample)
        out = ct.c_int()
        ok = _lib.ctorch_kernel_svm_predict(self._ptr, sample_mat._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("KernelSVM.predict failed")
        return int(out.value)

    def score(
        self,
        x_test: Matrix | Sequence[object],
        y_test: Matrix | Sequence[object],
    ) -> float:
        x_mat = _coerce_matrix(x_test)
        y_mat = _coerce_row_vector(y_test)
        out = ct.c_double()
        ok = _lib.ctorch_kernel_svm_score(self._ptr, x_mat._ptr, y_mat._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("KernelSVM.score failed")
        return float(out.value)

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _lib.ctorch_kernel_svm_destroy(ptr)
            self._ptr = ct.c_void_p()


class RandomFourierSVM:
    __slots__ = ("_ptr",)

    def __init__(
        self,
        x_train: Matrix | Sequence[object],
        y_train: Matrix | Sequence[object],
        d_features: int,
        gamma: float,
        learning_rate: float,
        max_iter: int,
        c_value: float,
    ) -> None:
        x_mat = _coerce_matrix(x_train)
        y_mat = _coerce_row_vector(y_train)
        ptr = _lib.ctorch_random_fourier_svm_create(
            x_mat._ptr,
            y_mat._ptr,
            int(d_features),
            float(gamma),
            float(learning_rate),
            int(max_iter),
            float(c_value),
        )
        if not ptr:
            raise _raise_last_error("RandomFourierSVM(...) failed")
        self._ptr = ct.c_void_p(ptr)

    def predict(self, sample: Matrix | Sequence[object]) -> int:
        sample_mat = _coerce_row_vector(sample)
        out = ct.c_int()
        ok = _lib.ctorch_random_fourier_svm_predict(self._ptr, sample_mat._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("RandomFourierSVM.predict failed")
        return int(out.value)

    def score(
        self,
        x_test: Matrix | Sequence[object],
        y_test: Matrix | Sequence[object],
    ) -> float:
        x_mat = _coerce_matrix(x_test)
        y_mat = _coerce_row_vector(y_test)
        out = ct.c_double()
        ok = _lib.ctorch_random_fourier_svm_score(self._ptr, x_mat._ptr, y_mat._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("RandomFourierSVM.score failed")
        return float(out.value)

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _lib.ctorch_random_fourier_svm_destroy(ptr)
            self._ptr = ct.c_void_p()


class GaussianNB:
    __slots__ = ("_ptr",)

    def __init__(
        self,
        x_train: Matrix | Sequence[object],
        y_train: Matrix | Sequence[object],
    ) -> None:
        x_mat = _coerce_matrix(x_train)
        y_mat = _coerce_row_vector(y_train)
        ptr = _lib.ctorch_gaussian_nb_create(x_mat._ptr, y_mat._ptr)
        if not ptr:
            raise _raise_last_error("GaussianNB(...) failed")
        self._ptr = ct.c_void_p(ptr)

    def predict(self, sample: Matrix | Sequence[object]) -> int:
        sample_mat = _coerce_row_vector(sample)
        out = ct.c_int()
        ok = _lib.ctorch_gaussian_nb_predict(self._ptr, sample_mat._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("GaussianNB.predict failed")
        return int(out.value)

    def score(
        self,
        x_test: Matrix | Sequence[object],
        y_test: Matrix | Sequence[object],
    ) -> float:
        x_mat = _coerce_matrix(x_test)
        y_mat = _coerce_row_vector(y_test)
        out = ct.c_double()
        ok = _lib.ctorch_gaussian_nb_score(self._ptr, x_mat._ptr, y_mat._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("GaussianNB.score failed")
        return float(out.value)

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _lib.ctorch_gaussian_nb_destroy(ptr)
            self._ptr = ct.c_void_p()


class KMeans:
    __slots__ = ("_ptr",)

    def __init__(self, k: int, x_train: Matrix | Sequence[object], max_iter: int = 100) -> None:
        x_mat = _coerce_matrix(x_train)
        ptr = _lib.ctorch_kmeans_create(int(k), x_mat._ptr, int(max_iter))
        if not ptr:
            raise _raise_last_error("KMeans(...) failed")
        self._ptr = ct.c_void_p(ptr)

    def get_assignments(self) -> list[int]:
        count = ct.c_size_t()
        ok = _lib.ctorch_kmeans_assignment_count(self._ptr, ct.byref(count))
        if not ok:
            raise _raise_last_error("KMeans.get_assignments failed")

        n = int(count.value)
        if n == 0:
            ok = _lib.ctorch_kmeans_get_assignments(self._ptr, None, 0)
            if not ok:
                raise _raise_last_error("KMeans.get_assignments failed")
            return []

        out = (ct.c_int * n)()
        ok = _lib.ctorch_kmeans_get_assignments(self._ptr, out, n)
        if not ok:
            raise _raise_last_error("KMeans.get_assignments failed")
        return [int(v) for v in out]

    @property
    def assignments(self) -> list[int]:
        return self.get_assignments()

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _lib.ctorch_kmeans_destroy(ptr)
            self._ptr = ct.c_void_p()


class PCA:
    __slots__ = ("_ptr",)

    def __init__(self, centered_x: Matrix | Sequence[object]) -> None:
        x_mat = _coerce_matrix(centered_x)
        ptr = _lib.ctorch_pca_create(x_mat._ptr)
        if not ptr:
            raise _raise_last_error("PCA(...) failed")
        self._ptr = ct.c_void_p(ptr)

    def compute_projection(
        self,
        k: int,
        max_iter: int = 1000,
        tol: float = 1e-9,
    ) -> Matrix:
        ptr = _lib.ctorch_pca_compute_projection(
            self._ptr,
            int(k),
            int(max_iter),
            float(tol),
        )
        if not ptr:
            raise _raise_last_error("PCA.compute_projection failed")
        return Matrix(_ptr=ptr)

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _lib.ctorch_pca_destroy(ptr)
            self._ptr = ct.c_void_p()

