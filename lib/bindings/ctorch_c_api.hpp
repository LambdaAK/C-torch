#pragma once

#ifndef CTORCH_C_API_HPP
#define CTORCH_C_API_HPP

#include <cstddef>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CTorchMatrix CTorchMatrix;
typedef struct CTorchKNN CTorchKNN;
typedef struct CTorchLinearRegression CTorchLinearRegression;
typedef struct CTorchLogisticRegression CTorchLogisticRegression;
typedef struct CTorchPerceptron CTorchPerceptron;
typedef struct CTorchSVM CTorchSVM;
typedef struct CTorchKernelSVM CTorchKernelSVM;
typedef struct CTorchRandomFourierSVM CTorchRandomFourierSVM;
typedef struct CTorchGaussianNB CTorchGaussianNB;
typedef struct CTorchKMeans CTorchKMeans;
typedef struct CTorchPCA CTorchPCA;
typedef struct CTorchMAB CTorchMAB;
typedef struct CTorchUCB CTorchUCB;

typedef enum CTorchOptimType
{
    CTORCH_OPTIM_GD = 0,
    CTORCH_OPTIM_SGD = 1,
    CTORCH_OPTIM_ADAGRAD = 2,
    CTORCH_OPTIM_RMSPROP = 3,
    CTORCH_OPTIM_ADAM = 4,
    CTORCH_OPTIM_ADAMW = 5
} CTorchOptimType;

typedef enum CTorchDataAugmentationType
{
    CTORCH_AUGMENT_NO_OP = 0,
    CTORCH_AUGMENT_POLY_2 = 1,
    CTORCH_AUGMENT_POLY_3 = 2,
    CTORCH_AUGMENT_POLY_4 = 3,
    CTORCH_AUGMENT_POLY_5 = 4,
    CTORCH_AUGMENT_RFF = 5
} CTorchDataAugmentationType;

typedef enum CTorchKernelType
{
    CTORCH_KERNEL_LINEAR = 0,
    CTORCH_KERNEL_POLYNOMIAL_2 = 1,
    CTORCH_KERNEL_POLYNOMIAL_3 = 2,
    CTORCH_KERNEL_RADIAL_BASIS = 3
} CTorchKernelType;

/**
 * Returns the last error message produced by the C API on the current thread.
 * The returned pointer remains valid until the next API call on the same thread.
 */
const char* ctorch_last_error(void);

/** Clears the thread-local error string. */
void ctorch_clear_error(void);

CTorchMatrix* ctorch_matrix_create(size_t rows, size_t cols);
CTorchMatrix* ctorch_matrix_from_array(
    size_t rows,
    size_t cols,
    const double* values,
    size_t value_count);
void ctorch_matrix_destroy(CTorchMatrix* matrix);

size_t ctorch_matrix_rows(const CTorchMatrix* matrix);
size_t ctorch_matrix_cols(const CTorchMatrix* matrix);

bool ctorch_matrix_get(
    const CTorchMatrix* matrix,
    size_t row,
    size_t col,
    double* out_value);
bool ctorch_matrix_set(
    CTorchMatrix* matrix,
    size_t row,
    size_t col,
    double value);

bool ctorch_matrix_to_array(
    const CTorchMatrix* matrix,
    double* out_values,
    size_t out_count);

CTorchMatrix* ctorch_matrix_add(const CTorchMatrix* lhs, const CTorchMatrix* rhs);
CTorchMatrix* ctorch_matrix_sub(const CTorchMatrix* lhs, const CTorchMatrix* rhs);
CTorchMatrix* ctorch_matrix_mul_scalar(const CTorchMatrix* matrix, double scalar);
CTorchMatrix* ctorch_matrix_matmul(const CTorchMatrix* lhs, const CTorchMatrix* rhs);
CTorchMatrix* ctorch_matrix_transpose(const CTorchMatrix* matrix);

CTorchKNN* ctorch_knn_create(size_t k, const CTorchMatrix* x_train, const CTorchMatrix* y_train);
void ctorch_knn_destroy(CTorchKNN* model);

bool ctorch_knn_predict(const CTorchKNN* model, const CTorchMatrix* sample, int* out_label);
bool ctorch_knn_score(CTorchKNN* model, const CTorchMatrix* x_test, const CTorchMatrix* y_test, double* out_score);

bool ctorch_knn_get_k(const CTorchKNN* model, size_t* out_k);
bool ctorch_knn_set_k(CTorchKNN* model, size_t new_k);

CTorchLinearRegression* ctorch_linear_regression_create(
    const CTorchMatrix* x_train,
    const CTorchMatrix* y_train,
    double learning_rate,
    int max_iter);
void ctorch_linear_regression_destroy(CTorchLinearRegression* model);
bool ctorch_linear_regression_predict(
    const CTorchLinearRegression* model,
    const CTorchMatrix* sample,
    double* out_value);
bool ctorch_linear_regression_score(
    const CTorchLinearRegression* model,
    const CTorchMatrix* x_test,
    const CTorchMatrix* y_test,
    double threshold,
    double* out_score);

CTorchLogisticRegression* ctorch_logistic_regression_create(
    const CTorchMatrix* x_train,
    const CTorchMatrix* y_train,
    CTorchOptimType optim_type,
    double learning_rate,
    int max_iter,
    int batch_size,
    double beta1,
    double beta2,
    double epsilon,
    double rho,
    double weight_decay,
    CTorchDataAugmentationType augmentation_type);
void ctorch_logistic_regression_destroy(CTorchLogisticRegression* model);
bool ctorch_logistic_regression_predict(
    const CTorchLogisticRegression* model,
    const CTorchMatrix* sample,
    int* out_label);
bool ctorch_logistic_regression_score(
    const CTorchLogisticRegression* model,
    const CTorchMatrix* x_test,
    const CTorchMatrix* y_test,
    double* out_score);

CTorchPerceptron* ctorch_perceptron_create(
    const CTorchMatrix* x_train,
    const CTorchMatrix* y_train,
    int epochs);
void ctorch_perceptron_destroy(CTorchPerceptron* model);
bool ctorch_perceptron_predict(
    const CTorchPerceptron* model,
    const CTorchMatrix* sample,
    int* out_label);
bool ctorch_perceptron_score(
    const CTorchPerceptron* model,
    const CTorchMatrix* x_test,
    const CTorchMatrix* y_test,
    double* out_score);

CTorchSVM* ctorch_svm_create(
    const CTorchMatrix* x_train,
    const CTorchMatrix* y_train,
    double learning_rate,
    int max_iter,
    double c_value,
    CTorchDataAugmentationType augmentation_type);
void ctorch_svm_destroy(CTorchSVM* model);
bool ctorch_svm_predict(
    const CTorchSVM* model,
    const CTorchMatrix* sample,
    int* out_label);
bool ctorch_svm_score(
    const CTorchSVM* model,
    const CTorchMatrix* x_test,
    const CTorchMatrix* y_test,
    double* out_score);

CTorchKernelSVM* ctorch_kernel_svm_create(
    const CTorchMatrix* x_train,
    const CTorchMatrix* y_train,
    double learning_rate,
    int max_iter,
    double c_value,
    CTorchKernelType kernel_type,
    double gamma);
void ctorch_kernel_svm_destroy(CTorchKernelSVM* model);
bool ctorch_kernel_svm_predict(
    const CTorchKernelSVM* model,
    const CTorchMatrix* sample,
    int* out_label);
bool ctorch_kernel_svm_score(
    const CTorchKernelSVM* model,
    const CTorchMatrix* x_test,
    const CTorchMatrix* y_test,
    double* out_score);

CTorchRandomFourierSVM* ctorch_random_fourier_svm_create(
    const CTorchMatrix* x_train,
    const CTorchMatrix* y_train,
    int d_features,
    double gamma,
    double learning_rate,
    int max_iter,
    double c_value);
void ctorch_random_fourier_svm_destroy(CTorchRandomFourierSVM* model);
bool ctorch_random_fourier_svm_predict(
    const CTorchRandomFourierSVM* model,
    const CTorchMatrix* sample,
    int* out_label);
bool ctorch_random_fourier_svm_score(
    const CTorchRandomFourierSVM* model,
    const CTorchMatrix* x_test,
    const CTorchMatrix* y_test,
    double* out_score);

CTorchGaussianNB* ctorch_gaussian_nb_create(
    const CTorchMatrix* x_train,
    const CTorchMatrix* y_train);
void ctorch_gaussian_nb_destroy(CTorchGaussianNB* model);
bool ctorch_gaussian_nb_predict(
    const CTorchGaussianNB* model,
    const CTorchMatrix* sample,
    int* out_label);
bool ctorch_gaussian_nb_score(
    CTorchGaussianNB* model,
    const CTorchMatrix* x_test,
    const CTorchMatrix* y_test,
    double* out_score);

CTorchKMeans* ctorch_kmeans_create(
    int k,
    const CTorchMatrix* x_train,
    int max_iter);
void ctorch_kmeans_destroy(CTorchKMeans* model);
bool ctorch_kmeans_assignment_count(
    const CTorchKMeans* model,
    size_t* out_count);
bool ctorch_kmeans_get_assignments(
    const CTorchKMeans* model,
    int* out_assignments,
    size_t out_count);

CTorchPCA* ctorch_pca_create(const CTorchMatrix* centered_x);
void ctorch_pca_destroy(CTorchPCA* model);
CTorchMatrix* ctorch_pca_compute_projection(
    CTorchPCA* model,
    int k,
    int max_iter,
    double tol);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CTORCH_C_API_HPP
