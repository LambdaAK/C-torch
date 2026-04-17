#include "bindings/ctorch_c_api.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "math/dataaugmentor.hpp"
#include "math/matrix.hpp"
#include "math/optim.hpp"
#include "ml/gaussian_nb.hpp"
#include "ml/kernelsvm.hpp"
#include "ml/kmeans.hpp"
#include "ml/knn.hpp"
#include "ml/linearregression.hpp"
#include "ml/logisticregression.hpp"
#include "ml/mab.hpp"
#include "ml/pca.hpp"
#include "ml/perceptron.hpp"
#include "ml/randomfouriersvm.hpp"
#include "ml/svm.hpp"
#include "ml/ucb.hpp"

namespace {

thread_local std::string g_last_error;

void set_error(std::string message)
{
    g_last_error = std::move(message);
}

void clear_error()
{
    g_last_error.clear();
}

template <typename T, typename Fn>
T run_api(Fn&& fn, T fallback)
{
    try
    {
        clear_error();
        return fn();
    }
    catch (const std::exception& ex)
    {
        set_error(ex.what());
        return fallback;
    }
    catch (...)
    {
        set_error("unknown C++ exception");
        return fallback;
    }
}

math::OptimType as_optim_type(CTorchOptimType optim_type)
{
    switch (optim_type)
    {
    case CTORCH_OPTIM_GD:
        return math::OptimType::GD;
    case CTORCH_OPTIM_SGD:
        return math::OptimType::SGD;
    case CTORCH_OPTIM_ADAGRAD:
        return math::OptimType::ADAGRAD;
    case CTORCH_OPTIM_RMSPROP:
        return math::OptimType::RMSPROP;
    case CTORCH_OPTIM_ADAM:
        return math::OptimType::ADAM;
    case CTORCH_OPTIM_ADAMW:
        return math::OptimType::ADAMW;
    default:
        throw std::invalid_argument("invalid optimizer type");
    }
}

DataAugmentationType as_augmentation_type(CTorchDataAugmentationType augmentation_type)
{
    switch (augmentation_type)
    {
    case CTORCH_AUGMENT_NO_OP:
        return DataAugmentationType::NO_OP;
    case CTORCH_AUGMENT_POLY_2:
        return DataAugmentationType::POLY_2;
    case CTORCH_AUGMENT_POLY_3:
        return DataAugmentationType::POLY_3;
    case CTORCH_AUGMENT_POLY_4:
        return DataAugmentationType::POLY_4;
    case CTORCH_AUGMENT_POLY_5:
        return DataAugmentationType::POLY_5;
    case CTORCH_AUGMENT_RFF:
        return DataAugmentationType::RFF;
    default:
        throw std::invalid_argument("invalid data augmentation type");
    }
}

ml::KernelOptions as_kernel_options(CTorchKernelType kernel_type, double gamma)
{
    switch (kernel_type)
    {
    case CTORCH_KERNEL_LINEAR:
        return ml::KernelOptions::linear();
    case CTORCH_KERNEL_POLYNOMIAL_2:
        return ml::KernelOptions::polynomial_2();
    case CTORCH_KERNEL_POLYNOMIAL_3:
        return ml::KernelOptions::polynomial_3();
    case CTORCH_KERNEL_RADIAL_BASIS:
        return ml::KernelOptions::radial_basis(gamma);
    default:
        throw std::invalid_argument("invalid kernel type");
    }
}

} // namespace

struct CTorchMatrix
{
    Matrix value;
};

struct CTorchKNN
{
    std::unique_ptr<ml::KNN> value;
};

struct CTorchLinearRegression
{
    std::unique_ptr<ml::LinearRegression> value;
};

struct CTorchLogisticRegression
{
    std::unique_ptr<ml::LogisticRegression> value;
};

struct CTorchPerceptron
{
    std::unique_ptr<ml::Perceptron> value;
};

struct CTorchSVM
{
    std::unique_ptr<ml::SVM> value;
};

struct CTorchKernelSVM
{
    std::unique_ptr<ml::KernelSVM> value;
};

struct CTorchRandomFourierSVM
{
    std::unique_ptr<ml::RandomFourierSVM> value;
};

struct CTorchGaussianNB
{
    std::unique_ptr<ml::GaussianNB> value;
};

struct CTorchKMeans
{
    std::unique_ptr<ml::KMeans> value;
};

struct CTorchPCA
{
    std::unique_ptr<ml::PCA> value;
};

struct CTorchMAB
{
    std::unique_ptr<ml::MAB> value;
};

struct CTorchUCB
{
    std::unique_ptr<ml::UCB> value;
};

namespace {

const Matrix& as_matrix_ref(const CTorchMatrix* handle)
{
    if (handle == nullptr)
    {
        throw std::invalid_argument("matrix handle is null");
    }
    return handle->value;
}

Matrix& as_matrix_mut(CTorchMatrix* handle)
{
    if (handle == nullptr)
    {
        throw std::invalid_argument("matrix handle is null");
    }
    return handle->value;
}

const ml::KNN& as_knn_ref(const CTorchKNN* handle)
{
    if (handle == nullptr || handle->value == nullptr)
    {
        throw std::invalid_argument("knn handle is null");
    }
    return *handle->value;
}

ml::KNN& as_knn_mut(CTorchKNN* handle)
{
    if (handle == nullptr || handle->value == nullptr)
    {
        throw std::invalid_argument("knn handle is null");
    }
    return *handle->value;
}

const ml::LinearRegression& as_linear_regression_ref(const CTorchLinearRegression* handle)
{
    if (handle == nullptr || handle->value == nullptr)
    {
        throw std::invalid_argument("linear regression handle is null");
    }
    return *handle->value;
}

const ml::LogisticRegression& as_logistic_regression_ref(const CTorchLogisticRegression* handle)
{
    if (handle == nullptr || handle->value == nullptr)
    {
        throw std::invalid_argument("logistic regression handle is null");
    }
    return *handle->value;
}

const ml::Perceptron& as_perceptron_ref(const CTorchPerceptron* handle)
{
    if (handle == nullptr || handle->value == nullptr)
    {
        throw std::invalid_argument("perceptron handle is null");
    }
    return *handle->value;
}

const ml::SVM& as_svm_ref(const CTorchSVM* handle)
{
    if (handle == nullptr || handle->value == nullptr)
    {
        throw std::invalid_argument("svm handle is null");
    }
    return *handle->value;
}

const ml::KernelSVM& as_kernel_svm_ref(const CTorchKernelSVM* handle)
{
    if (handle == nullptr || handle->value == nullptr)
    {
        throw std::invalid_argument("kernel svm handle is null");
    }
    return *handle->value;
}

const ml::RandomFourierSVM& as_random_fourier_svm_ref(const CTorchRandomFourierSVM* handle)
{
    if (handle == nullptr || handle->value == nullptr)
    {
        throw std::invalid_argument("random fourier svm handle is null");
    }
    return *handle->value;
}

const ml::GaussianNB& as_gaussian_nb_ref(const CTorchGaussianNB* handle)
{
    if (handle == nullptr || handle->value == nullptr)
    {
        throw std::invalid_argument("gaussian nb handle is null");
    }
    return *handle->value;
}

ml::GaussianNB& as_gaussian_nb_mut(CTorchGaussianNB* handle)
{
    if (handle == nullptr || handle->value == nullptr)
    {
        throw std::invalid_argument("gaussian nb handle is null");
    }
    return *handle->value;
}

const ml::KMeans& as_kmeans_ref(const CTorchKMeans* handle)
{
    if (handle == nullptr || handle->value == nullptr)
    {
        throw std::invalid_argument("kmeans handle is null");
    }
    return *handle->value;
}

ml::PCA& as_pca_mut(CTorchPCA* handle)
{
    if (handle == nullptr || handle->value == nullptr)
    {
        throw std::invalid_argument("pca handle is null");
    }
    return *handle->value;
}

ml::MAB& as_mab_mut(CTorchMAB* handle)
{
    if (handle == nullptr || handle->value == nullptr)
    {
        throw std::invalid_argument("mab handle is null");
    }
    return *handle->value;
}

ml::UCB& as_ucb_mut(CTorchUCB* handle)
{
    if (handle == nullptr || handle->value == nullptr)
    {
        throw std::invalid_argument("ucb handle is null");
    }
    return *handle->value;
}

} // namespace

extern "C" {

const char* ctorch_last_error(void)
{
    return g_last_error.c_str();
}

void ctorch_clear_error(void)
{
    clear_error();
}

CTorchMatrix* ctorch_matrix_create(size_t rows, size_t cols)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        auto handle = std::make_unique<CTorchMatrix>();
        handle->value = Matrix(rows, cols);
        return handle.release();
    }, nullptr);
}

CTorchMatrix* ctorch_matrix_from_array(
    size_t rows,
    size_t cols,
    const double* values,
    size_t value_count)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        const size_t expected = rows * cols;
        if (value_count != expected)
        {
            throw std::invalid_argument("value_count does not match rows * cols");
        }
        if (expected > 0 && values == nullptr)
        {
            throw std::invalid_argument("values pointer is null");
        }

        auto handle = std::make_unique<CTorchMatrix>();
        handle->value = Matrix(rows, cols);
        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t c = 0; c < cols; ++c)
            {
                handle->value(r, c) = values[r * cols + c];
            }
        }
        return handle.release();
    }, nullptr);
}

void ctorch_matrix_destroy(CTorchMatrix* matrix)
{
    delete matrix;
}

size_t ctorch_matrix_rows(const CTorchMatrix* matrix)
{
    return run_api<size_t>([&]() -> size_t {
        return as_matrix_ref(matrix).numRows();
    }, static_cast<size_t>(0));
}

size_t ctorch_matrix_cols(const CTorchMatrix* matrix)
{
    return run_api<size_t>([&]() -> size_t {
        return as_matrix_ref(matrix).numCols();
    }, static_cast<size_t>(0));
}

bool ctorch_matrix_get(
    const CTorchMatrix* matrix,
    size_t row,
    size_t col,
    double* out_value)
{
    return run_api<bool>([&]() -> bool {
        if (out_value == nullptr)
        {
            throw std::invalid_argument("out_value pointer is null");
        }
        *out_value = as_matrix_ref(matrix).at(row, col);
        return true;
    }, false);
}

bool ctorch_matrix_set(
    CTorchMatrix* matrix,
    size_t row,
    size_t col,
    double value)
{
    return run_api<bool>([&]() -> bool {
        as_matrix_mut(matrix)(row, col) = value;
        return true;
    }, false);
}

bool ctorch_matrix_to_array(
    const CTorchMatrix* matrix,
    double* out_values,
    size_t out_count)
{
    return run_api<bool>([&]() -> bool {
        const Matrix& m = as_matrix_ref(matrix);
        const size_t expected = m.numRows() * m.numCols();
        if (out_count != expected)
        {
            throw std::invalid_argument("out_count does not match matrix size");
        }
        if (expected > 0 && out_values == nullptr)
        {
            throw std::invalid_argument("out_values pointer is null");
        }
        for (size_t r = 0; r < m.numRows(); ++r)
        {
            for (size_t c = 0; c < m.numCols(); ++c)
            {
                out_values[r * m.numCols() + c] = m(r, c);
            }
        }
        return true;
    }, false);
}

CTorchMatrix* ctorch_matrix_add(const CTorchMatrix* lhs, const CTorchMatrix* rhs)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        auto out = std::make_unique<CTorchMatrix>();
        out->value = as_matrix_ref(lhs) + as_matrix_ref(rhs);
        return out.release();
    }, nullptr);
}

CTorchMatrix* ctorch_matrix_sub(const CTorchMatrix* lhs, const CTorchMatrix* rhs)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        auto out = std::make_unique<CTorchMatrix>();
        out->value = as_matrix_ref(lhs) - as_matrix_ref(rhs);
        return out.release();
    }, nullptr);
}

CTorchMatrix* ctorch_matrix_mul_scalar(const CTorchMatrix* matrix, double scalar)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        auto out = std::make_unique<CTorchMatrix>();
        out->value = as_matrix_ref(matrix) * scalar;
        return out.release();
    }, nullptr);
}

CTorchMatrix* ctorch_matrix_matmul(const CTorchMatrix* lhs, const CTorchMatrix* rhs)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        auto out = std::make_unique<CTorchMatrix>();
        out->value = as_matrix_ref(lhs) * as_matrix_ref(rhs);
        return out.release();
    }, nullptr);
}

CTorchMatrix* ctorch_matrix_transpose(const CTorchMatrix* matrix)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        auto out = std::make_unique<CTorchMatrix>();
        out->value = as_matrix_ref(matrix).transpose();
        return out.release();
    }, nullptr);
}

CTorchKNN* ctorch_knn_create(
    size_t k,
    const CTorchMatrix* x_train,
    const CTorchMatrix* y_train)
{
    return run_api<CTorchKNN*>([&]() -> CTorchKNN* {
        auto handle = std::make_unique<CTorchKNN>();
        handle->value = std::make_unique<ml::KNN>(
            k,
            as_matrix_ref(x_train),
            as_matrix_ref(y_train));
        return handle.release();
    }, nullptr);
}

void ctorch_knn_destroy(CTorchKNN* model)
{
    delete model;
}

bool ctorch_knn_predict(
    const CTorchKNN* model,
    const CTorchMatrix* sample,
    int* out_label)
{
    return run_api<bool>([&]() -> bool {
        if (out_label == nullptr)
        {
            throw std::invalid_argument("out_label pointer is null");
        }
        *out_label = as_knn_ref(model).predict(as_matrix_ref(sample));
        return true;
    }, false);
}

bool ctorch_knn_score(
    CTorchKNN* model,
    const CTorchMatrix* x_test,
    const CTorchMatrix* y_test,
    double* out_score)
{
    return run_api<bool>([&]() -> bool {
        if (out_score == nullptr)
        {
            throw std::invalid_argument("out_score pointer is null");
        }
        *out_score = as_knn_mut(model).score(as_matrix_ref(x_test), as_matrix_ref(y_test));
        return true;
    }, false);
}

bool ctorch_knn_get_k(const CTorchKNN* model, size_t* out_k)
{
    return run_api<bool>([&]() -> bool {
        if (out_k == nullptr)
        {
            throw std::invalid_argument("out_k pointer is null");
        }
        *out_k = as_knn_ref(model).getK();
        return true;
    }, false);
}

bool ctorch_knn_set_k(CTorchKNN* model, size_t new_k)
{
    return run_api<bool>([&]() -> bool {
        as_knn_mut(model).setK(new_k);
        return true;
    }, false);
}

CTorchLinearRegression* ctorch_linear_regression_create(
    const CTorchMatrix* x_train,
    const CTorchMatrix* y_train,
    double learning_rate,
    int max_iter)
{
    return run_api<CTorchLinearRegression*>([&]() -> CTorchLinearRegression* {
        auto handle = std::make_unique<CTorchLinearRegression>();
        handle->value = std::make_unique<ml::LinearRegression>(
            as_matrix_ref(x_train),
            as_matrix_ref(y_train),
            learning_rate,
            max_iter);
        return handle.release();
    }, nullptr);
}

void ctorch_linear_regression_destroy(CTorchLinearRegression* model)
{
    delete model;
}

bool ctorch_linear_regression_predict(
    const CTorchLinearRegression* model,
    const CTorchMatrix* sample,
    double* out_value)
{
    return run_api<bool>([&]() -> bool {
        if (out_value == nullptr)
        {
            throw std::invalid_argument("out_value pointer is null");
        }
        *out_value = as_linear_regression_ref(model).predict(as_matrix_ref(sample));
        return true;
    }, false);
}

bool ctorch_linear_regression_score(
    const CTorchLinearRegression* model,
    const CTorchMatrix* x_test,
    const CTorchMatrix* y_test,
    double threshold,
    double* out_score)
{
    return run_api<bool>([&]() -> bool {
        if (out_score == nullptr)
        {
            throw std::invalid_argument("out_score pointer is null");
        }
        *out_score = as_linear_regression_ref(model).score(
            as_matrix_ref(x_test),
            as_matrix_ref(y_test),
            threshold);
        return true;
    }, false);
}

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
    CTorchDataAugmentationType augmentation_type)
{
    return run_api<CTorchLogisticRegression*>([&]() -> CTorchLogisticRegression* {
        math::OptimParams params(
            as_optim_type(optim_type),
            learning_rate,
            max_iter,
            Matrix(),
            Matrix(),
            batch_size,
            beta1,
            beta2,
            epsilon,
            rho,
            weight_decay);

        auto handle = std::make_unique<CTorchLogisticRegression>();
        handle->value = std::make_unique<ml::LogisticRegression>(
            as_matrix_ref(x_train),
            as_matrix_ref(y_train),
            params,
            as_augmentation_type(augmentation_type));
        return handle.release();
    }, nullptr);
}

void ctorch_logistic_regression_destroy(CTorchLogisticRegression* model)
{
    delete model;
}

bool ctorch_logistic_regression_predict(
    const CTorchLogisticRegression* model,
    const CTorchMatrix* sample,
    int* out_label)
{
    return run_api<bool>([&]() -> bool {
        if (out_label == nullptr)
        {
            throw std::invalid_argument("out_label pointer is null");
        }
        *out_label = static_cast<int>(as_logistic_regression_ref(model).predict(as_matrix_ref(sample)));
        return true;
    }, false);
}

bool ctorch_logistic_regression_score(
    const CTorchLogisticRegression* model,
    const CTorchMatrix* x_test,
    const CTorchMatrix* y_test,
    double* out_score)
{
    return run_api<bool>([&]() -> bool {
        if (out_score == nullptr)
        {
            throw std::invalid_argument("out_score pointer is null");
        }
        *out_score = as_logistic_regression_ref(model).score(
            as_matrix_ref(x_test),
            as_matrix_ref(y_test));
        return true;
    }, false);
}

CTorchPerceptron* ctorch_perceptron_create(
    const CTorchMatrix* x_train,
    const CTorchMatrix* y_train,
    int epochs)
{
    return run_api<CTorchPerceptron*>([&]() -> CTorchPerceptron* {
        auto handle = std::make_unique<CTorchPerceptron>();
        handle->value = std::make_unique<ml::Perceptron>(
            as_matrix_ref(x_train),
            as_matrix_ref(y_train),
            epochs);
        return handle.release();
    }, nullptr);
}

void ctorch_perceptron_destroy(CTorchPerceptron* model)
{
    delete model;
}

bool ctorch_perceptron_predict(
    const CTorchPerceptron* model,
    const CTorchMatrix* sample,
    int* out_label)
{
    return run_api<bool>([&]() -> bool {
        if (out_label == nullptr)
        {
            throw std::invalid_argument("out_label pointer is null");
        }
        *out_label = as_perceptron_ref(model).predict(as_matrix_ref(sample));
        return true;
    }, false);
}

bool ctorch_perceptron_score(
    const CTorchPerceptron* model,
    const CTorchMatrix* x_test,
    const CTorchMatrix* y_test,
    double* out_score)
{
    return run_api<bool>([&]() -> bool {
        if (out_score == nullptr)
        {
            throw std::invalid_argument("out_score pointer is null");
        }
        *out_score = as_perceptron_ref(model).score(
            as_matrix_ref(x_test),
            as_matrix_ref(y_test));
        return true;
    }, false);
}

CTorchSVM* ctorch_svm_create(
    const CTorchMatrix* x_train,
    const CTorchMatrix* y_train,
    double learning_rate,
    int max_iter,
    double c_value,
    CTorchDataAugmentationType augmentation_type)
{
    return run_api<CTorchSVM*>([&]() -> CTorchSVM* {
        auto handle = std::make_unique<CTorchSVM>();
        handle->value = std::make_unique<ml::SVM>(
            as_matrix_ref(x_train),
            as_matrix_ref(y_train),
            learning_rate,
            max_iter,
            c_value,
            as_augmentation_type(augmentation_type));
        return handle.release();
    }, nullptr);
}

void ctorch_svm_destroy(CTorchSVM* model)
{
    delete model;
}

bool ctorch_svm_predict(
    const CTorchSVM* model,
    const CTorchMatrix* sample,
    int* out_label)
{
    return run_api<bool>([&]() -> bool {
        if (out_label == nullptr)
        {
            throw std::invalid_argument("out_label pointer is null");
        }
        *out_label = as_svm_ref(model).predict(as_matrix_ref(sample));
        return true;
    }, false);
}

bool ctorch_svm_score(
    const CTorchSVM* model,
    const CTorchMatrix* x_test,
    const CTorchMatrix* y_test,
    double* out_score)
{
    return run_api<bool>([&]() -> bool {
        if (out_score == nullptr)
        {
            throw std::invalid_argument("out_score pointer is null");
        }
        *out_score = as_svm_ref(model).score(
            as_matrix_ref(x_test),
            as_matrix_ref(y_test));
        return true;
    }, false);
}

CTorchKernelSVM* ctorch_kernel_svm_create(
    const CTorchMatrix* x_train,
    const CTorchMatrix* y_train,
    double learning_rate,
    int max_iter,
    double c_value,
    CTorchKernelType kernel_type,
    double gamma)
{
    return run_api<CTorchKernelSVM*>([&]() -> CTorchKernelSVM* {
        auto handle = std::make_unique<CTorchKernelSVM>();
        handle->value = std::make_unique<ml::KernelSVM>(
            as_matrix_ref(x_train),
            as_matrix_ref(y_train),
            learning_rate,
            max_iter,
            c_value,
            as_kernel_options(kernel_type, gamma));
        return handle.release();
    }, nullptr);
}

void ctorch_kernel_svm_destroy(CTorchKernelSVM* model)
{
    delete model;
}

bool ctorch_kernel_svm_predict(
    const CTorchKernelSVM* model,
    const CTorchMatrix* sample,
    int* out_label)
{
    return run_api<bool>([&]() -> bool {
        if (out_label == nullptr)
        {
            throw std::invalid_argument("out_label pointer is null");
        }
        *out_label = as_kernel_svm_ref(model).predict(as_matrix_ref(sample));
        return true;
    }, false);
}

bool ctorch_kernel_svm_score(
    const CTorchKernelSVM* model,
    const CTorchMatrix* x_test,
    const CTorchMatrix* y_test,
    double* out_score)
{
    return run_api<bool>([&]() -> bool {
        if (out_score == nullptr)
        {
            throw std::invalid_argument("out_score pointer is null");
        }
        *out_score = as_kernel_svm_ref(model).score(
            as_matrix_ref(x_test),
            as_matrix_ref(y_test));
        return true;
    }, false);
}

CTorchRandomFourierSVM* ctorch_random_fourier_svm_create(
    const CTorchMatrix* x_train,
    const CTorchMatrix* y_train,
    int d_features,
    double gamma,
    double learning_rate,
    int max_iter,
    double c_value)
{
    return run_api<CTorchRandomFourierSVM*>([&]() -> CTorchRandomFourierSVM* {
        auto handle = std::make_unique<CTorchRandomFourierSVM>();
        handle->value = std::make_unique<ml::RandomFourierSVM>(
            as_matrix_ref(x_train),
            as_matrix_ref(y_train),
            d_features,
            gamma,
            learning_rate,
            max_iter,
            c_value);
        return handle.release();
    }, nullptr);
}

void ctorch_random_fourier_svm_destroy(CTorchRandomFourierSVM* model)
{
    delete model;
}

bool ctorch_random_fourier_svm_predict(
    const CTorchRandomFourierSVM* model,
    const CTorchMatrix* sample,
    int* out_label)
{
    return run_api<bool>([&]() -> bool {
        if (out_label == nullptr)
        {
            throw std::invalid_argument("out_label pointer is null");
        }
        *out_label = as_random_fourier_svm_ref(model).predict(as_matrix_ref(sample));
        return true;
    }, false);
}

bool ctorch_random_fourier_svm_score(
    const CTorchRandomFourierSVM* model,
    const CTorchMatrix* x_test,
    const CTorchMatrix* y_test,
    double* out_score)
{
    return run_api<bool>([&]() -> bool {
        if (out_score == nullptr)
        {
            throw std::invalid_argument("out_score pointer is null");
        }
        *out_score = as_random_fourier_svm_ref(model).score(
            as_matrix_ref(x_test),
            as_matrix_ref(y_test));
        return true;
    }, false);
}

CTorchGaussianNB* ctorch_gaussian_nb_create(
    const CTorchMatrix* x_train,
    const CTorchMatrix* y_train)
{
    return run_api<CTorchGaussianNB*>([&]() -> CTorchGaussianNB* {
        auto handle = std::make_unique<CTorchGaussianNB>();
        handle->value = std::make_unique<ml::GaussianNB>(
            as_matrix_ref(x_train),
            as_matrix_ref(y_train));
        return handle.release();
    }, nullptr);
}

void ctorch_gaussian_nb_destroy(CTorchGaussianNB* model)
{
    delete model;
}

bool ctorch_gaussian_nb_predict(
    const CTorchGaussianNB* model,
    const CTorchMatrix* sample,
    int* out_label)
{
    return run_api<bool>([&]() -> bool {
        if (out_label == nullptr)
        {
            throw std::invalid_argument("out_label pointer is null");
        }
        *out_label = as_gaussian_nb_ref(model).predict(as_matrix_ref(sample));
        return true;
    }, false);
}

bool ctorch_gaussian_nb_score(
    CTorchGaussianNB* model,
    const CTorchMatrix* x_test,
    const CTorchMatrix* y_test,
    double* out_score)
{
    return run_api<bool>([&]() -> bool {
        if (out_score == nullptr)
        {
            throw std::invalid_argument("out_score pointer is null");
        }
        *out_score = as_gaussian_nb_mut(model).score(
            as_matrix_ref(x_test),
            as_matrix_ref(y_test));
        return true;
    }, false);
}

CTorchKMeans* ctorch_kmeans_create(
    int k,
    const CTorchMatrix* x_train,
    int max_iter)
{
    return run_api<CTorchKMeans*>([&]() -> CTorchKMeans* {
        auto handle = std::make_unique<CTorchKMeans>();
        handle->value = std::make_unique<ml::KMeans>(
            k,
            as_matrix_ref(x_train),
            max_iter);
        return handle.release();
    }, nullptr);
}

void ctorch_kmeans_destroy(CTorchKMeans* model)
{
    delete model;
}

bool ctorch_kmeans_assignment_count(
    const CTorchKMeans* model,
    size_t* out_count)
{
    return run_api<bool>([&]() -> bool {
        if (out_count == nullptr)
        {
            throw std::invalid_argument("out_count pointer is null");
        }
        const std::vector<int> assignments = as_kmeans_ref(model).getAssignments();
        *out_count = assignments.size();
        return true;
    }, false);
}

bool ctorch_kmeans_get_assignments(
    const CTorchKMeans* model,
    int* out_assignments,
    size_t out_count)
{
    return run_api<bool>([&]() -> bool {
        const std::vector<int> assignments = as_kmeans_ref(model).getAssignments();
        if (out_count != assignments.size())
        {
            throw std::invalid_argument("out_count does not match assignment count");
        }
        if (out_count > 0 && out_assignments == nullptr)
        {
            throw std::invalid_argument("out_assignments pointer is null");
        }
        for (size_t i = 0; i < out_count; ++i)
        {
            out_assignments[i] = assignments[i];
        }
        return true;
    }, false);
}

CTorchPCA* ctorch_pca_create(const CTorchMatrix* centered_x)
{
    return run_api<CTorchPCA*>([&]() -> CTorchPCA* {
        auto handle = std::make_unique<CTorchPCA>();
        handle->value = std::make_unique<ml::PCA>(as_matrix_ref(centered_x));
        return handle.release();
    }, nullptr);
}

void ctorch_pca_destroy(CTorchPCA* model)
{
    delete model;
}

CTorchMatrix* ctorch_pca_compute_projection(
    CTorchPCA* model,
    int k,
    int max_iter,
    double tol)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        auto out = std::make_unique<CTorchMatrix>();
        out->value = as_pca_mut(model).compute_projection_mat(k, max_iter, static_cast<float>(tol));
        return out.release();
    }, nullptr);
}


} // extern "C"
