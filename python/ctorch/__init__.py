from __future__ import annotations

import ctypes as ct
import os
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    "MAB",
    "UCB",
    "Sequential",
    "NNOptimizer",
    "OptimType",
    "DataAugmentationType",
    "KernelType",
    "NNOptimType",
    "Expr",
    "SymbolicLoss",
    "symbolic_optimize",
    "encode_regression_label",
    "regression_label_row",
    "LinearProgramSense",
    "LinearProgramResult",
    "solve_linear_program",
    "QuadraticProgramResult",
    "solve_quadratic_program",
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


class NNOptimType(IntEnum):
    SGD = 0
    ADAGRAD = 1
    RMSPROP = 2
    ADAM = 3
    ADAMW = 4


class LinearProgramSense(IntEnum):
    MINIMIZE = 0
    MAXIMIZE = 1


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
_lib.ctorch_matrix_equals.argtypes = [ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_bool)]
_lib.ctorch_matrix_equals.restype = ct.c_bool

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

_lib.ctorch_mab_create.argtypes = [ct.c_int, ct.c_float]
_lib.ctorch_mab_create.restype = ct.c_void_p
_lib.ctorch_mab_destroy.argtypes = [ct.c_void_p]
_lib.ctorch_mab_destroy.restype = None
_lib.ctorch_mab_select_arm.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
_lib.ctorch_mab_select_arm.restype = ct.c_bool
_lib.ctorch_mab_update.argtypes = [ct.c_void_p, ct.c_int, ct.c_double]
_lib.ctorch_mab_update.restype = ct.c_bool
_lib.ctorch_mab_set_epsilon.argtypes = [ct.c_void_p, ct.c_float]
_lib.ctorch_mab_set_epsilon.restype = ct.c_bool

_lib.ctorch_ucb_create.argtypes = [ct.c_int]
_lib.ctorch_ucb_create.restype = ct.c_void_p
_lib.ctorch_ucb_destroy.argtypes = [ct.c_void_p]
_lib.ctorch_ucb_destroy.restype = None
_lib.ctorch_ucb_select_arm.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
_lib.ctorch_ucb_select_arm.restype = ct.c_bool
_lib.ctorch_ucb_update.argtypes = [ct.c_void_p, ct.c_int, ct.c_double]
_lib.ctorch_ucb_update.restype = ct.c_bool

_lib.ctorch_sequential_create.argtypes = []
_lib.ctorch_sequential_create.restype = ct.c_void_p
_lib.ctorch_sequential_destroy.argtypes = [ct.c_void_p]
_lib.ctorch_sequential_destroy.restype = None
_lib.ctorch_sequential_add_linear.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]
_lib.ctorch_sequential_add_linear.restype = ct.c_bool
_lib.ctorch_sequential_add_relu.argtypes = [ct.c_void_p]
_lib.ctorch_sequential_add_relu.restype = ct.c_bool
_lib.ctorch_sequential_add_sigmoid.argtypes = [ct.c_void_p]
_lib.ctorch_sequential_add_sigmoid.restype = ct.c_bool
_lib.ctorch_sequential_add_tanh.argtypes = [ct.c_void_p]
_lib.ctorch_sequential_add_tanh.restype = ct.c_bool
_lib.ctorch_sequential_forward.argtypes = [ct.c_void_p, ct.c_void_p]
_lib.ctorch_sequential_forward.restype = ct.c_void_p
_lib.ctorch_sequential_backward.argtypes = [ct.c_void_p, ct.c_void_p]
_lib.ctorch_sequential_backward.restype = ct.c_bool
_lib.ctorch_sequential_save.argtypes = [ct.c_void_p, ct.c_char_p]
_lib.ctorch_sequential_save.restype = ct.c_bool
_lib.ctorch_sequential_load.argtypes = [ct.c_void_p, ct.c_char_p]
_lib.ctorch_sequential_load.restype = ct.c_bool

_lib.ctorch_nn_optimizer_create.argtypes = [
    ct.c_void_p,
    ct.c_int,
    ct.c_float,
    ct.c_size_t,
    ct.c_float,
    ct.c_float,
    ct.c_float,
    ct.c_float,
    ct.c_float,
]
_lib.ctorch_nn_optimizer_create.restype = ct.c_void_p
_lib.ctorch_nn_optimizer_destroy.argtypes = [ct.c_void_p]
_lib.ctorch_nn_optimizer_destroy.restype = None
_lib.ctorch_nn_optimizer_zero_grad.argtypes = [ct.c_void_p]
_lib.ctorch_nn_optimizer_zero_grad.restype = ct.c_bool
_lib.ctorch_nn_optimizer_step.argtypes = [ct.c_void_p]
_lib.ctorch_nn_optimizer_step.restype = ct.c_bool

_lib.ctorch_matrix_eye.argtypes = [ct.c_size_t]
_lib.ctorch_matrix_eye.restype = ct.c_void_p
_lib.ctorch_matrix_row.argtypes = [ct.c_void_p, ct.c_size_t]
_lib.ctorch_matrix_row.restype = ct.c_void_p
_lib.ctorch_matrix_l2_norm_cols.argtypes = [ct.c_void_p]
_lib.ctorch_matrix_l2_norm_cols.restype = ct.c_void_p
_lib.ctorch_matrix_center_cols.argtypes = [ct.c_void_p]
_lib.ctorch_matrix_center_cols.restype = ct.c_void_p
_lib.ctorch_matrix_euclidean_distance.argtypes = [ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_double)]
_lib.ctorch_matrix_euclidean_distance.restype = ct.c_bool
_lib.ctorch_matrix_inner_product.argtypes = [ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_double)]
_lib.ctorch_matrix_inner_product.restype = ct.c_bool
_lib.ctorch_matrix_relu.argtypes = [ct.c_void_p]
_lib.ctorch_matrix_relu.restype = ct.c_void_p
_lib.ctorch_matrix_relu_deriv.argtypes = [ct.c_void_p]
_lib.ctorch_matrix_relu_deriv.restype = ct.c_void_p
_lib.ctorch_matrix_sigmoid.argtypes = [ct.c_void_p]
_lib.ctorch_matrix_sigmoid.restype = ct.c_void_p
_lib.ctorch_matrix_sigmoid_deriv.argtypes = [ct.c_void_p]
_lib.ctorch_matrix_sigmoid_deriv.restype = ct.c_void_p
_lib.ctorch_matrix_tanh.argtypes = [ct.c_void_p]
_lib.ctorch_matrix_tanh.restype = ct.c_void_p
_lib.ctorch_matrix_tanh_deriv.argtypes = [ct.c_void_p]
_lib.ctorch_matrix_tanh_deriv.restype = ct.c_void_p
_lib.ctorch_matrix_elm_wise_product.argtypes = [ct.c_void_p, ct.c_void_p]
_lib.ctorch_matrix_elm_wise_product.restype = ct.c_void_p

_lib.ctorch_expr_num.argtypes = [ct.c_double]
_lib.ctorch_expr_num.restype = ct.c_void_p
_lib.ctorch_expr_var.argtypes = [ct.c_char_p]
_lib.ctorch_expr_var.restype = ct.c_void_p
for _name in (
    "ctorch_expr_add",
    "ctorch_expr_sub",
    "ctorch_expr_mul",
    "ctorch_expr_div",
    "ctorch_expr_pow",
):
    getattr(_lib, _name).argtypes = [ct.c_void_p, ct.c_void_p]
    getattr(_lib, _name).restype = ct.c_void_p
_lib.ctorch_expr_neg.argtypes = [ct.c_void_p]
_lib.ctorch_expr_neg.restype = ct.c_void_p
for _name in (
    "ctorch_expr_exp",
    "ctorch_expr_log",
    "ctorch_expr_sqrt",
    "ctorch_expr_abs",
    "ctorch_expr_sigmoid",
):
    getattr(_lib, _name).argtypes = [ct.c_void_p]
    getattr(_lib, _name).restype = ct.c_void_p
_lib.ctorch_expr_max.argtypes = [ct.c_void_p, ct.c_void_p]
_lib.ctorch_expr_max.restype = ct.c_void_p
_lib.ctorch_expr_min.argtypes = [ct.c_void_p, ct.c_void_p]
_lib.ctorch_expr_min.restype = ct.c_void_p
_lib.ctorch_expr_destroy.argtypes = [ct.c_void_p]
_lib.ctorch_expr_destroy.restype = None
_lib.ctorch_expr_to_string.argtypes = [ct.c_void_p]
_lib.ctorch_expr_to_string.restype = ct.c_char_p
_lib.ctorch_expr_simplify.argtypes = [ct.c_void_p]
_lib.ctorch_expr_simplify.restype = ct.c_void_p
_lib.ctorch_expr_substitute.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_void_p]
_lib.ctorch_expr_substitute.restype = ct.c_void_p
_lib.ctorch_expr_variable_count.argtypes = [ct.c_void_p]
_lib.ctorch_expr_variable_count.restype = ct.c_size_t
_lib.ctorch_expr_variable_name.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_char_p, ct.c_size_t]
_lib.ctorch_expr_variable_name.restype = ct.c_bool
_lib.ctorch_expr_evaluate.argtypes = [ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_char_p), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
_lib.ctorch_expr_evaluate.restype = ct.c_bool
_lib.ctorch_expr_gradient.argtypes = [ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_char_p), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
_lib.ctorch_expr_gradient.restype = ct.c_bool

_lib.ctorch_loss_regression_mse_create.argtypes = [ct.c_int, ct.c_double]
_lib.ctorch_loss_regression_mse_create.restype = ct.c_void_p
_lib.ctorch_loss_logistic_create.argtypes = [ct.c_int]
_lib.ctorch_loss_logistic_create.restype = ct.c_void_p
_lib.ctorch_loss_destroy.argtypes = [ct.c_void_p]
_lib.ctorch_loss_destroy.restype = None

_lib.ctorch_symbolic_optimize.argtypes = [
    ct.c_void_p,
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
]
_lib.ctorch_symbolic_optimize.restype = ct.c_void_p
_lib.ctorch_param_map_destroy.argtypes = [ct.c_void_p]
_lib.ctorch_param_map_destroy.restype = None
_lib.ctorch_param_map_size.argtypes = [ct.c_void_p]
_lib.ctorch_param_map_size.restype = ct.c_size_t
_lib.ctorch_param_map_get.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_char_p, ct.c_size_t, ct.POINTER(ct.c_double)]
_lib.ctorch_param_map_get.restype = ct.c_bool

_lib.ctorch_linear_program_solve.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_double),
    ct.c_size_t,
    ct.POINTER(ct.c_double),
    ct.c_size_t,
    ct.c_int,
    ct.POINTER(ct.c_double),
    ct.c_size_t,
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_bool),
    ct.POINTER(ct.c_bool),
    ct.POINTER(ct.c_int),
]
_lib.ctorch_linear_program_solve.restype = ct.c_bool

_lib.ctorch_quadratic_program_solve.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_double),
    ct.c_size_t,
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_double),
    ct.c_size_t,
    ct.c_double,
    ct.POINTER(ct.c_double),
    ct.c_size_t,
    ct.POINTER(ct.c_double),
    ct.c_size_t,
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_bool),
    ct.POINTER(ct.c_int),
]
_lib.ctorch_quadratic_program_solve.restype = ct.c_bool


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


def _c_str_array(strings: Sequence[str]) -> tuple[Any, list[bytes]]:
    encoded = [s.encode("utf-8") for s in strings]
    arr = (ct.c_char_p * len(encoded))(*encoded)
    return arr, encoded


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

    @classmethod
    def eye(cls, dim: int) -> Matrix:
        ptr = _lib.ctorch_matrix_eye(ct.c_size_t(int(dim)))
        if not ptr:
            raise _raise_last_error("Matrix.eye failed")
        return cls(_ptr=ptr)

    def row(self, index: int) -> Matrix:
        ptr = _lib.ctorch_matrix_row(self._ptr, ct.c_size_t(int(index)))
        if not ptr:
            raise _raise_last_error("Matrix.row failed")
        return Matrix(_ptr=ptr)

    def rows(self) -> list[Matrix]:
        return [self.row(i) for i in range(self.num_rows)]

    def l2_norm_cols(self) -> Matrix:
        ptr = _lib.ctorch_matrix_l2_norm_cols(self._ptr)
        if not ptr:
            raise _raise_last_error("Matrix.l2_norm_cols failed")
        return Matrix(_ptr=ptr)

    def center_cols(self) -> Matrix:
        ptr = _lib.ctorch_matrix_center_cols(self._ptr)
        if not ptr:
            raise _raise_last_error("Matrix.center_cols failed")
        return Matrix(_ptr=ptr)

    def euclidean_distance(self, other: Matrix | Sequence[object]) -> float:
        rhs = _coerce_matrix(other)
        out = ct.c_double()
        ok = _lib.ctorch_matrix_euclidean_distance(self._ptr, rhs._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("Matrix.euclidean_distance failed")
        return float(out.value)

    def inner_product(self, other: Matrix | Sequence[object]) -> float:
        rhs = _coerce_matrix(other)
        out = ct.c_double()
        ok = _lib.ctorch_matrix_inner_product(self._ptr, rhs._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("Matrix.inner_product failed")
        return float(out.value)

    def relu(self) -> Matrix:
        ptr = _lib.ctorch_matrix_relu(self._ptr)
        if not ptr:
            raise _raise_last_error("Matrix.relu failed")
        return Matrix(_ptr=ptr)

    def relu_deriv(self) -> Matrix:
        ptr = _lib.ctorch_matrix_relu_deriv(self._ptr)
        if not ptr:
            raise _raise_last_error("Matrix.relu_deriv failed")
        return Matrix(_ptr=ptr)

    def sigmoid(self) -> Matrix:
        ptr = _lib.ctorch_matrix_sigmoid(self._ptr)
        if not ptr:
            raise _raise_last_error("Matrix.sigmoid failed")
        return Matrix(_ptr=ptr)

    def sigmoid_deriv(self) -> Matrix:
        ptr = _lib.ctorch_matrix_sigmoid_deriv(self._ptr)
        if not ptr:
            raise _raise_last_error("Matrix.sigmoid_deriv failed")
        return Matrix(_ptr=ptr)

    def tanh(self) -> Matrix:
        ptr = _lib.ctorch_matrix_tanh(self._ptr)
        if not ptr:
            raise _raise_last_error("Matrix.tanh failed")
        return Matrix(_ptr=ptr)

    def tanh_deriv(self) -> Matrix:
        ptr = _lib.ctorch_matrix_tanh_deriv(self._ptr)
        if not ptr:
            raise _raise_last_error("Matrix.tanh_deriv failed")
        return Matrix(_ptr=ptr)

    def elementwise_mul(self, other: Matrix | Sequence[object]) -> Matrix:
        rhs = _coerce_matrix(other)
        ptr = _lib.ctorch_matrix_elm_wise_product(self._ptr, rhs._ptr)
        if not ptr:
            raise _raise_last_error("Matrix.elementwise_mul failed")
        return Matrix(_ptr=ptr)

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        out = ct.c_bool()
        ok = _lib.ctorch_matrix_equals(self._ptr, other._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("Matrix.__eq__ failed")
        return bool(out.value)

    def __str__(self) -> str:
        row_count, col_count = self.shape
        if row_count == 0 or col_count == 0:
            return f"Matrix({row_count}x{col_count}) []"

        max_rows = 4
        max_cols = 6

        def _fmt(value: float) -> str:
            return f"{value:.6g}"

        rendered_rows: list[str] = []
        preview_rows = min(row_count, max_rows)
        preview_cols = min(col_count, max_cols)
        for i in range(preview_rows):
            rendered = ", ".join(_fmt(self.get(i, j)) for j in range(preview_cols))
            if col_count > max_cols:
                rendered = f"{rendered}, ..."
            rendered_rows.append(f"[{rendered}]")
        if row_count > max_rows:
            rendered_rows.append("...")
        return f"Matrix({row_count}x{col_count}) [{', '.join(rendered_rows)}]"

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


class MAB:
    __slots__ = ("_ptr",)

    def __init__(self, n_arms: int, eps: float) -> None:
        ptr = _lib.ctorch_mab_create(int(n_arms), float(eps))
        if not ptr:
            raise _raise_last_error("MAB(...) failed")
        self._ptr = ct.c_void_p(ptr)

    def select_arm(self) -> int:
        out = ct.c_int()
        ok = _lib.ctorch_mab_select_arm(self._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("MAB.select_arm failed")
        return int(out.value)

    def update(self, arm: int, reward: float) -> None:
        ok = _lib.ctorch_mab_update(self._ptr, int(arm), float(reward))
        if not ok:
            raise _raise_last_error("MAB.update failed")

    def set_epsilon(self, eps: float) -> None:
        ok = _lib.ctorch_mab_set_epsilon(self._ptr, float(eps))
        if not ok:
            raise _raise_last_error("MAB.set_epsilon failed")

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _lib.ctorch_mab_destroy(ptr)
            self._ptr = ct.c_void_p()


class UCB:
    __slots__ = ("_ptr",)

    def __init__(self, n_arms: int) -> None:
        ptr = _lib.ctorch_ucb_create(int(n_arms))
        if not ptr:
            raise _raise_last_error("UCB(...) failed")
        self._ptr = ct.c_void_p(ptr)

    def select_arm(self) -> int:
        out = ct.c_int()
        ok = _lib.ctorch_ucb_select_arm(self._ptr, ct.byref(out))
        if not ok:
            raise _raise_last_error("UCB.select_arm failed")
        return int(out.value)

    def update(self, arm: int, reward: float) -> None:
        ok = _lib.ctorch_ucb_update(self._ptr, int(arm), float(reward))
        if not ok:
            raise _raise_last_error("UCB.update failed")

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _lib.ctorch_ucb_destroy(ptr)
            self._ptr = ct.c_void_p()


class Sequential:
    """Feed-forward stack of linear and activation layers (matches C++ alternating layout)."""

    __slots__ = ("_ptr",)

    def __init__(self, _ptr: int | None = None) -> None:
        if _ptr is not None:
            self._ptr = ct.c_void_p(_ptr)
            return
        ptr = _lib.ctorch_sequential_create()
        if not ptr:
            raise _raise_last_error("Sequential() failed")
        self._ptr = ct.c_void_p(ptr)

    def add_linear(self, input_dim: int, output_dim: int) -> Sequential:
        ok = _lib.ctorch_sequential_add_linear(self._ptr, int(input_dim), int(output_dim))
        if not ok:
            raise _raise_last_error("Sequential.add_linear failed")
        return self

    def add_relu(self) -> Sequential:
        ok = _lib.ctorch_sequential_add_relu(self._ptr)
        if not ok:
            raise _raise_last_error("Sequential.add_relu failed")
        return self

    def add_sigmoid(self) -> Sequential:
        ok = _lib.ctorch_sequential_add_sigmoid(self._ptr)
        if not ok:
            raise _raise_last_error("Sequential.add_sigmoid failed")
        return self

    def add_tanh(self) -> Sequential:
        ok = _lib.ctorch_sequential_add_tanh(self._ptr)
        if not ok:
            raise _raise_last_error("Sequential.add_tanh failed")
        return self

    def forward(self, x: Matrix | Sequence[object]) -> Matrix:
        """Run a forward pass.

        ``x`` must follow the C++ layout: ``input_dim`` rows and **one column**
        per sample (a column vector of shape ``(input_dim, 1)``). To process a
        mini-batch, call ``forward`` once per sample (and ``backward`` per
        sample) before ``NNOptimizer.step``, matching ``batch_size``.
        """
        x_mat = _coerce_matrix(x)
        ptr = _lib.ctorch_sequential_forward(self._ptr, x_mat._ptr)
        if not ptr:
            raise _raise_last_error("Sequential.forward failed")
        return Matrix(_ptr=ptr)

    def backward(self, dL_d_output: Matrix | Sequence[object]) -> None:
        grad_mat = _coerce_matrix(dL_d_output)
        ok = _lib.ctorch_sequential_backward(self._ptr, grad_mat._ptr)
        if not ok:
            raise _raise_last_error("Sequential.backward failed")

    def save(self, filepath: str) -> None:
        encoded = filepath.encode("utf-8")
        ok = _lib.ctorch_sequential_save(self._ptr, encoded)
        if not ok:
            raise _raise_last_error("Sequential.save failed")

    def load(self, filepath: str) -> None:
        encoded = filepath.encode("utf-8")
        ok = _lib.ctorch_sequential_load(self._ptr, encoded)
        if not ok:
            raise _raise_last_error("Sequential.load failed")

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _lib.ctorch_sequential_destroy(ptr)
            self._ptr = ct.c_void_p()


class NNOptimizer:
    __slots__ = ("_ptr", "_model")

    def __init__(
        self,
        model: Sequential,
        optim_type: NNOptimType | int = NNOptimType.SGD,
        learning_rate: float = 0.01,
        batch_size: int = 1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        rho: float = 0.99,
        weight_decay: float = 0.0,
    ) -> None:
        self._model = model
        optim = _coerce_enum(optim_type, NNOptimType, "optim_type")
        ptr = _lib.ctorch_nn_optimizer_create(
            model._ptr,
            int(optim),
            float(learning_rate),
            ct.c_size_t(int(batch_size)),
            float(beta1),
            float(beta2),
            float(epsilon),
            float(rho),
            float(weight_decay),
        )
        if not ptr:
            raise _raise_last_error("NNOptimizer(...) failed")
        self._ptr = ct.c_void_p(ptr)

    def zero_grad(self) -> None:
        ok = _lib.ctorch_nn_optimizer_zero_grad(self._ptr)
        if not ok:
            raise _raise_last_error("NNOptimizer.zero_grad failed")

    def step(self) -> None:
        ok = _lib.ctorch_nn_optimizer_step(self._ptr)
        if not ok:
            raise _raise_last_error("NNOptimizer.step failed")

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _lib.ctorch_nn_optimizer_destroy(ptr)
            self._ptr = ct.c_void_p()


def encode_regression_label(y: float) -> int:
    """Encode a real-valued regression target for :class:`SymbolicLoss` MSE (internal scale 1e4)."""
    return int(round(float(y) * 10000.0))


def regression_label_row(values: Sequence[float]) -> Matrix:
    """Build a ``1 × N`` label row matrix for regression MSE training from real targets."""
    enc = [encode_regression_label(v) for v in values]
    return Matrix([enc])


class Expr:
    """Symbolic scalar expression (AST). Use ``Expr.num`` / ``Expr.var`` and Python operators."""

    __slots__ = ("_ptr",)

    def __init__(self, ptr: int) -> None:
        self._ptr = ct.c_void_p(ptr)

    @staticmethod
    def num(value: float) -> Expr:
        ptr = _lib.ctorch_expr_num(float(value))
        if not ptr:
            raise _raise_last_error("Expr.num failed")
        return Expr(ptr)

    @staticmethod
    def var(name: str) -> Expr:
        b = name.encode("utf-8")
        ptr = _lib.ctorch_expr_var(b)
        if not ptr:
            raise _raise_last_error("Expr.var failed")
        return Expr(ptr)

    @staticmethod
    def _bin(op: Any, lhs: Expr, rhs: Expr) -> Expr:
        ptr = op(lhs._ptr, rhs._ptr)
        if not ptr:
            raise _raise_last_error("expression build failed")
        return Expr(ptr)

    def __add__(self, other: Expr) -> Expr:
        return Expr._bin(_lib.ctorch_expr_add, self, other)

    def __sub__(self, other: Expr) -> Expr:
        return Expr._bin(_lib.ctorch_expr_sub, self, other)

    def __mul__(self, other: Expr) -> Expr:
        return Expr._bin(_lib.ctorch_expr_mul, self, other)

    def __truediv__(self, other: Expr) -> Expr:
        return Expr._bin(_lib.ctorch_expr_div, self, other)

    def __pow__(self, other: Expr) -> Expr:
        return Expr._bin(_lib.ctorch_expr_pow, self, other)

    def __neg__(self) -> Expr:
        ptr = _lib.ctorch_expr_neg(self._ptr)
        if not ptr:
            raise _raise_last_error("Expr.__neg__ failed")
        return Expr(ptr)

    def exp(self) -> Expr:
        ptr = _lib.ctorch_expr_exp(self._ptr)
        if not ptr:
            raise _raise_last_error("Expr.exp failed")
        return Expr(ptr)

    def log(self) -> Expr:
        ptr = _lib.ctorch_expr_log(self._ptr)
        if not ptr:
            raise _raise_last_error("Expr.log failed")
        return Expr(ptr)

    def sqrt(self) -> Expr:
        ptr = _lib.ctorch_expr_sqrt(self._ptr)
        if not ptr:
            raise _raise_last_error("Expr.sqrt failed")
        return Expr(ptr)

    def abs(self) -> Expr:
        ptr = _lib.ctorch_expr_abs(self._ptr)
        if not ptr:
            raise _raise_last_error("Expr.abs failed")
        return Expr(ptr)

    def sigmoid(self) -> Expr:
        ptr = _lib.ctorch_expr_sigmoid(self._ptr)
        if not ptr:
            raise _raise_last_error("Expr.sigmoid failed")
        return Expr(ptr)

    def max(self, other: Expr) -> Expr:
        return Expr._bin(_lib.ctorch_expr_max, self, other)

    def min(self, other: Expr) -> Expr:
        return Expr._bin(_lib.ctorch_expr_min, self, other)

    def simplify(self) -> Expr:
        ptr = _lib.ctorch_expr_simplify(self._ptr)
        if not ptr:
            raise _raise_last_error("Expr.simplify failed")
        return Expr(ptr)

    def substitute(self, name: str, replacement: Expr) -> Expr:
        nb = name.encode("utf-8")
        ptr = _lib.ctorch_expr_substitute(self._ptr, nb, replacement._ptr)
        if not ptr:
            raise _raise_last_error("Expr.substitute failed")
        return Expr(ptr)

    def variables(self) -> list[str]:
        n = int(_lib.ctorch_expr_variable_count(self._ptr))
        names: list[str] = []
        buf = ct.create_string_buffer(256)
        for i in range(n):
            ok = _lib.ctorch_expr_variable_name(self._ptr, ct.c_size_t(i), buf, ct.c_size_t(256))
            if not ok:
                raise _raise_last_error("Expr.variables failed")
            names.append(buf.value.decode("utf-8"))
        return names

    def evaluate(self, bindings: Mapping[str, float]) -> float:
        keys = list(bindings.keys())
        arr, _keep = _c_str_array(keys)
        vals = (ct.c_double * len(keys))(*[float(bindings[k]) for k in keys])
        out = ct.c_double()
        ok = _lib.ctorch_expr_evaluate(self._ptr, ct.c_size_t(len(keys)), arr, vals, ct.byref(out))
        if not ok:
            raise _raise_last_error("Expr.evaluate failed")
        return float(out.value)

    def gradient(self, bindings: Mapping[str, float]) -> dict[str, float]:
        keys = list(bindings.keys())
        arr, _keep = _c_str_array(keys)
        vals = (ct.c_double * len(keys))(*[float(bindings[k]) for k in keys])
        outs = (ct.c_double * len(keys))()
        ok = _lib.ctorch_expr_gradient(self._ptr, ct.c_size_t(len(keys)), arr, vals, outs)
        if not ok:
            raise _raise_last_error("Expr.gradient failed")
        return {keys[i]: float(outs[i]) for i in range(len(keys))}

    def __str__(self) -> str:
        raw = _lib.ctorch_expr_to_string(self._ptr)
        if raw is None:
            return "<Expr ?>"
        return raw.decode("utf-8", errors="replace")

    def __repr__(self) -> str:
        return f"Expr({str(self)!r})"

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _lib.ctorch_expr_destroy(ptr)
            self._ptr = ct.c_void_p()


class SymbolicLoss:
    """Symbolic supervised loss for :func:`symbolic_optimize` (linear regression MSE or logistic)."""

    __slots__ = ("_ptr",)

    def __init__(self, ptr: int) -> None:
        self._ptr = ct.c_void_p(ptr)

    @staticmethod
    def regression_mse(feature_dim: int, *, l2: float = 0.0) -> SymbolicLoss:
        ptr = _lib.ctorch_loss_regression_mse_create(int(feature_dim), float(l2))
        if not ptr:
            raise _raise_last_error("SymbolicLoss.regression_mse failed")
        return SymbolicLoss(ptr)

    @staticmethod
    def logistic(feature_dim: int) -> SymbolicLoss:
        ptr = _lib.ctorch_loss_logistic_create(int(feature_dim))
        if not ptr:
            raise _raise_last_error("SymbolicLoss.logistic failed")
        return SymbolicLoss(ptr)

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _lib.ctorch_loss_destroy(ptr)
            self._ptr = ct.c_void_p()


def symbolic_optimize(
    loss: SymbolicLoss,
    x_train: Matrix,
    y_train: Matrix,
    *,
    optim_type: OptimType | int = OptimType.ADAM,
    learning_rate: float = 0.05,
    max_iter: int = 800,
    batch_size: int = 32,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    rho: float = 0.99,
    weight_decay: float = 0.0,
) -> dict[str, float]:
    """Fit weights ``w0..w{d-1}`` and bias ``b`` using the math stack optimizers."""
    ot = _coerce_enum(optim_type, OptimType, "optim_type")
    pm = _lib.ctorch_symbolic_optimize(
        loss._ptr,
        x_train._ptr,
        y_train._ptr,
        int(ot),
        float(learning_rate),
        int(max_iter),
        int(batch_size),
        float(beta1),
        float(beta2),
        float(epsilon),
        float(rho),
        float(weight_decay),
    )
    if not pm:
        raise _raise_last_error("symbolic_optimize failed")
    try:
        n = int(_lib.ctorch_param_map_size(pm))
        out: dict[str, float] = {}
        buf = ct.create_string_buffer(256)
        for i in range(n):
            v = ct.c_double()
            ok = _lib.ctorch_param_map_get(pm, ct.c_size_t(i), buf, ct.c_size_t(256), ct.byref(v))
            if not ok:
                raise _raise_last_error("symbolic_optimize readback failed")
            out[buf.value.decode("utf-8")] = float(v.value)
        return out
    finally:
        _lib.ctorch_param_map_destroy(pm)


@dataclass(frozen=True)
class LinearProgramResult:
    solution: list[float]
    objective: float
    optimal: bool
    unbounded: bool
    iterations: int


def solve_linear_program(
    a: Matrix,
    b: Sequence[float],
    c: Sequence[float],
    *,
    sense: LinearProgramSense | int = LinearProgramSense.MINIMIZE,
) -> LinearProgramResult:
    """Solve ``min/max cᵀx`` s.t. ``A x ≤ b``, ``x ≥ 0`` (simplex backend)."""
    rows, cols = a.shape
    if len(b) != rows or len(c) != cols:
        raise ValueError("A shape must align with len(b) and len(c)")
    b_arr = (ct.c_double * len(b))(*[float(x) for x in b])
    c_arr = (ct.c_double * len(c))(*[float(x) for x in c])
    sol = (ct.c_double * cols)()
    obj = ct.c_double()
    opt = ct.c_bool()
    unb = ct.c_bool()
    iters = ct.c_int()
    s = _coerce_enum(sense, LinearProgramSense, "sense")
    ok = _lib.ctorch_linear_program_solve(
        a._ptr,
        b_arr,
        ct.c_size_t(len(b)),
        c_arr,
        ct.c_size_t(len(c)),
        int(s),
        sol,
        ct.c_size_t(cols),
        ct.byref(obj),
        ct.byref(opt),
        ct.byref(unb),
        ct.byref(iters),
    )
    if not ok:
        raise _raise_last_error("solve_linear_program failed")
    return LinearProgramResult(
        solution=[float(sol[i]) for i in range(cols)],
        objective=float(obj.value),
        optimal=bool(opt.value),
        unbounded=bool(unb.value),
        iterations=int(iters.value),
    )


@dataclass(frozen=True)
class QuadraticProgramResult:
    solution: list[float]
    objective: float
    converged: bool
    iterations: int


def solve_quadratic_program(
    q: Matrix,
    c: Sequence[float],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    *,
    equality_coeffs: Sequence[float] | None = None,
    equality_value: float = 0.0,
    initial: Sequence[float] | None = None,
) -> QuadraticProgramResult:
    """Bounded QP with optional equality ``aᵀx = value`` (projected gradient backend)."""
    n = q.num_rows
    if q.shape != (n, n) or len(c) != n or len(lower_bounds) != n or len(upper_bounds) != n:
        raise ValueError("inconsistent dimensions for QP")
    c_arr = (ct.c_double * n)(*[float(x) for x in c])
    lo = (ct.c_double * n)(*[float(x) for x in lower_bounds])
    hi = (ct.c_double * n)(*[float(x) for x in upper_bounds])
    eq_len = len(equality_coeffs) if equality_coeffs is not None else 0
    eq_arr = (ct.c_double * eq_len)(*map(float, equality_coeffs)) if eq_len else None
    init_len = len(initial) if initial is not None else 0
    init_arr = (ct.c_double * init_len)(*map(float, initial)) if init_len else None
    sol = (ct.c_double * n)()
    obj = ct.c_double()
    conv = ct.c_bool()
    iters = ct.c_int()
    ok = _lib.ctorch_quadratic_program_solve(
        q._ptr,
        c_arr,
        ct.c_size_t(n),
        lo,
        hi,
        eq_arr,
        ct.c_size_t(eq_len),
        float(equality_value),
        init_arr,
        ct.c_size_t(init_len),
        sol,
        ct.c_size_t(n),
        ct.byref(obj),
        ct.byref(conv),
        ct.byref(iters),
    )
    if not ok:
        raise _raise_last_error("solve_quadratic_program failed")
    return QuadraticProgramResult(
        solution=[float(sol[i]) for i in range(n)],
        objective=float(obj.value),
        converged=bool(conv.value),
        iterations=int(iters.value),
    )
