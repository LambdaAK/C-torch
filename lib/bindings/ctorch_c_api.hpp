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

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CTORCH_C_API_HPP

