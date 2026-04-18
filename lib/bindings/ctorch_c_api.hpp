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
typedef struct CTorchSequential CTorchSequential;
typedef struct CTorchNNOptimizer CTorchNNOptimizer;
typedef struct CTorchExpr CTorchExpr;
typedef struct CTorchLossFunction CTorchLossFunction;
typedef struct CTorchParamMap CTorchParamMap;

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

typedef enum CTorchNNOptimType
{
    CTORCH_NN_OPTIM_SGD = 0,
    CTORCH_NN_OPTIM_ADAGRAD = 1,
    CTORCH_NN_OPTIM_RMSPROP = 2,
    CTORCH_NN_OPTIM_ADAM = 3,
    CTORCH_NN_OPTIM_ADAMW = 4
} CTorchNNOptimType;

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
CTorchMatrix* ctorch_data_augment_no_op(const CTorchMatrix* x);
CTorchMatrix* ctorch_data_augment_poly_2(const CTorchMatrix* x);
CTorchMatrix* ctorch_data_augment_poly_3(const CTorchMatrix* x);
CTorchMatrix* ctorch_data_augment_poly_4(const CTorchMatrix* x);
CTorchMatrix* ctorch_data_augment_poly_5(const CTorchMatrix* x);
CTorchMatrix* ctorch_data_augment_rff(const CTorchMatrix* x, int d_features, double gamma);
CTorchMatrix* ctorch_data_augment_dispatch(
    const CTorchMatrix* x,
    CTorchDataAugmentationType augmentation,
    int d_features,
    double gamma);

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

CTorchMAB* ctorch_mab_create(int n_arms, float eps);
void ctorch_mab_destroy(CTorchMAB* model);
bool ctorch_mab_select_arm(CTorchMAB* model, int* out_arm);
bool ctorch_mab_update(CTorchMAB* model, int arm, double reward);
bool ctorch_mab_set_epsilon(CTorchMAB* model, float eps);

CTorchUCB* ctorch_ucb_create(int n_arms);
void ctorch_ucb_destroy(CTorchUCB* model);
bool ctorch_ucb_select_arm(CTorchUCB* model, int* out_arm);
bool ctorch_ucb_update(CTorchUCB* model, int arm, double reward);

CTorchSequential* ctorch_sequential_create(void);
void ctorch_sequential_destroy(CTorchSequential* model);
bool ctorch_sequential_add_linear(CTorchSequential* model, int input_dim, int output_dim);
bool ctorch_sequential_add_relu(CTorchSequential* model);
bool ctorch_sequential_add_sigmoid(CTorchSequential* model);
bool ctorch_sequential_add_tanh(CTorchSequential* model);
CTorchMatrix* ctorch_sequential_forward(CTorchSequential* model, const CTorchMatrix* x);
bool ctorch_sequential_backward(CTorchSequential* model, const CTorchMatrix* dL_d_output);
bool ctorch_sequential_save(CTorchSequential* model, const char* filepath);
bool ctorch_sequential_load(CTorchSequential* model, const char* filepath);

CTorchNNOptimizer* ctorch_nn_optimizer_create(
    CTorchSequential* model,
    CTorchNNOptimType optim_type,
    float learning_rate,
    size_t batch_size,
    float beta1,
    float beta2,
    float epsilon,
    float rho,
    float weight_decay);
void ctorch_nn_optimizer_destroy(CTorchNNOptimizer* optimizer);
bool ctorch_nn_optimizer_zero_grad(CTorchNNOptimizer* optimizer);
bool ctorch_nn_optimizer_step(CTorchNNOptimizer* optimizer);

CTorchMatrix* ctorch_matrix_eye(size_t dim);
CTorchMatrix* ctorch_matrix_row(const CTorchMatrix* matrix, size_t row);
CTorchMatrix* ctorch_matrix_l2_norm_cols(const CTorchMatrix* matrix);
CTorchMatrix* ctorch_matrix_center_cols(const CTorchMatrix* matrix);
bool ctorch_matrix_euclidean_distance(const CTorchMatrix* lhs, const CTorchMatrix* rhs, double* out_value);
bool ctorch_matrix_inner_product(const CTorchMatrix* lhs, const CTorchMatrix* rhs, double* out_value);
CTorchMatrix* ctorch_matrix_relu(const CTorchMatrix* matrix);
CTorchMatrix* ctorch_matrix_relu_deriv(const CTorchMatrix* matrix);
CTorchMatrix* ctorch_matrix_sigmoid(const CTorchMatrix* matrix);
CTorchMatrix* ctorch_matrix_sigmoid_deriv(const CTorchMatrix* matrix);
CTorchMatrix* ctorch_matrix_tanh(const CTorchMatrix* matrix);
CTorchMatrix* ctorch_matrix_tanh_deriv(const CTorchMatrix* matrix);
CTorchMatrix* ctorch_matrix_elm_wise_product(const CTorchMatrix* lhs, const CTorchMatrix* rhs);

CTorchExpr* ctorch_expr_num(double value);
CTorchExpr* ctorch_expr_var(const char* name);
CTorchExpr* ctorch_expr_add(const CTorchExpr* lhs, const CTorchExpr* rhs);
CTorchExpr* ctorch_expr_sub(const CTorchExpr* lhs, const CTorchExpr* rhs);
CTorchExpr* ctorch_expr_mul(const CTorchExpr* lhs, const CTorchExpr* rhs);
CTorchExpr* ctorch_expr_div(const CTorchExpr* lhs, const CTorchExpr* rhs);
CTorchExpr* ctorch_expr_pow(const CTorchExpr* lhs, const CTorchExpr* rhs);
CTorchExpr* ctorch_expr_neg(const CTorchExpr* operand);
CTorchExpr* ctorch_expr_exp(const CTorchExpr* operand);
CTorchExpr* ctorch_expr_log(const CTorchExpr* operand);
CTorchExpr* ctorch_expr_sqrt(const CTorchExpr* operand);
CTorchExpr* ctorch_expr_abs(const CTorchExpr* operand);
CTorchExpr* ctorch_expr_max(const CTorchExpr* lhs, const CTorchExpr* rhs);
CTorchExpr* ctorch_expr_min(const CTorchExpr* lhs, const CTorchExpr* rhs);
CTorchExpr* ctorch_expr_sigmoid(const CTorchExpr* operand);
void ctorch_expr_destroy(CTorchExpr* expr);
const char* ctorch_expr_to_string(const CTorchExpr* expr);
CTorchExpr* ctorch_expr_simplify(const CTorchExpr* expr);
CTorchExpr* ctorch_expr_substitute(const CTorchExpr* expr, const char* var_name, const CTorchExpr* replacement);
size_t ctorch_expr_variable_count(const CTorchExpr* expr);
bool ctorch_expr_variable_name(const CTorchExpr* expr, size_t index, char* out_buf, size_t out_buf_len);
bool ctorch_expr_evaluate(
    const CTorchExpr* expr,
    size_t pair_count,
    const char** var_names,
    const double* var_values,
    double* out_value);
bool ctorch_expr_gradient(
    const CTorchExpr* expr,
    size_t pair_count,
    const char** var_names,
    const double* var_values,
    double* out_partials);

CTorchLossFunction* ctorch_loss_regression_mse_create(int feature_dim, double l2_lambda);
CTorchLossFunction* ctorch_loss_logistic_create(int feature_dim);
void ctorch_loss_destroy(CTorchLossFunction* loss);

CTorchParamMap* ctorch_symbolic_optimize(
    const CTorchLossFunction* loss,
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
    double weight_decay);
void ctorch_param_map_destroy(CTorchParamMap* map);
size_t ctorch_param_map_size(const CTorchParamMap* map);
bool ctorch_param_map_get(const CTorchParamMap* map, size_t index, char* key_buf, size_t key_buf_len, double* out_value);

typedef enum CTorchLinearProgramSense
{
    CTORCH_LP_MINIMIZE = 0,
    CTORCH_LP_MAXIMIZE = 1
} CTorchLinearProgramSense;

bool ctorch_linear_program_solve(
    const CTorchMatrix* a,
    const double* b,
    size_t b_count,
    const double* c,
    size_t c_count,
    CTorchLinearProgramSense sense,
    double* solution_out,
    size_t solution_capacity,
    double* objective_out,
    bool* optimal_out,
    bool* unbounded_out,
    int* iterations_out);

bool ctorch_quadratic_program_solve(
    const CTorchMatrix* q,
    const double* c,
    size_t n,
    const double* lower_bounds,
    const double* upper_bounds,
    const double* equality_coeffs,
    size_t equality_len,
    double equality_value,
    const double* initial_solution,
    size_t initial_len,
    double* solution_out,
    size_t solution_capacity,
    double* objective_out,
    bool* converged_out,
    int* iterations_out);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CTORCH_C_API_HPP
