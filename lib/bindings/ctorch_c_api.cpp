#include "bindings/ctorch_c_api.hpp"

#include <cstring>
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
#include "ml/nn.hpp"
#include "ml/nn_optim.hpp"
#include "ml/pca.hpp"
#include "ml/perceptron.hpp"
#include "ml/randomfouriersvm.hpp"
#include "ml/svm.hpp"
#include "ml/ucb.hpp"

#include "math/ast.hpp"
#include "math/lossfunction.hpp"
#include "math/optim_base.hpp"
#include "math/optim_lp.hpp"
#include "math/optim_qp.hpp"
#include "math/optimizer.hpp"

#include <set>
#include <unordered_map>

namespace ml_bind_math {

class RegressionMSELoss final : public LossFunction
{
    size_t feature_dim;
    double l2_lambda;

public:
    RegressionMSELoss(size_t dim, double l2) : feature_dim(dim), l2_lambda(l2) {}

    std::shared_ptr<math::ASTNode> sample_loss(const Matrix& x, int y_scaled) const override
    {
        const double y = static_cast<double>(y_scaled) / 10000.0;
        std::shared_ptr<math::ASTNode> pred = math::Num(0.0);
        for (size_t i = 0; i < x.numCols(); ++i)
        {
            pred = pred + math::Var("w" + std::to_string(i)) * math::Num(x.at(0, i));
        }
        pred = pred + math::Var("b");
        auto err = pred - math::Num(y);
        return math::Num(0.5) * err * err;
    }

    std::shared_ptr<math::ASTNode> regularizer() const override
    {
        if (l2_lambda <= 0.0)
        {
            return math::Num(0.0);
        }
        std::shared_ptr<math::ASTNode> sum = math::Num(0.0);
        for (size_t i = 0; i < feature_dim; ++i)
        {
            auto wi = math::Var("w" + std::to_string(i));
            sum = sum + wi * wi;
        }
        auto b = math::Var("b");
        sum = sum + b * b;
        return math::Num(l2_lambda) * sum;
    }
};

class LogisticLossFn final : public LossFunction
{
public:
    std::shared_ptr<math::ASTNode> sample_loss(const Matrix& x, int y) const override
    {
        std::shared_ptr<math::ASTNode> w_transpose_x = math::Num(0.0);
        for (size_t i = 0; i < x.numCols(); ++i)
        {
            auto w_i = math::Var("w" + std::to_string(i));
            auto x_i = math::Num(x.at(0, i));
            w_transpose_x = w_transpose_x + w_i * x_i;
        }
        auto b = math::Var("b");
        auto z = w_transpose_x + b;
        auto y_hat = math::Sigmoid(z);
        return -math::Num(static_cast<double>(y)) * math::Log(y_hat) -
               (math::Num(1.0) - math::Num(static_cast<double>(y))) * math::Log(math::Num(1.0) - y_hat);
    }

    std::shared_ptr<math::ASTNode> regularizer() const override
    {
        return math::Num(0.0);
    }
};

std::unordered_map<std::string, double> zero_initial_theta(
    const LossFunction& loss,
    const Matrix& xTr,
    const Matrix& yTr)
{
    std::set<std::string> names;
    const auto rows = xTr.rowsAsMatrices();
    for (size_t i = 0; i < xTr.numRows(); ++i)
    {
        const int yi = static_cast<int>(yTr.at(0, i));
        auto root = loss.sample_loss(rows[i], yi);
        auto vi = root->variables();
        names.insert(vi.begin(), vi.end());
    }
    {
        auto r = loss.regularizer()->variables();
        names.insert(r.begin(), r.end());
    }
    std::unordered_map<std::string, double> theta;
    for (const auto& n : names)
    {
        theta[n] = 0.0;
    }
    return theta;
}

} // namespace ml_bind_math

namespace {

thread_local std::string g_last_error;
thread_local std::string g_expr_to_string_buf;

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

Matrix apply_data_augmentation(
    const Matrix& x,
    CTorchDataAugmentationType augmentation,
    int d_features,
    double gamma)
{
    switch (augmentation)
    {
    case CTORCH_AUGMENT_NO_OP:
        return DataAugmentor::no_op(x);
    case CTORCH_AUGMENT_POLY_2:
        return DataAugmentor::poly_2(x);
    case CTORCH_AUGMENT_POLY_3:
        return DataAugmentor::poly_3(x);
    case CTORCH_AUGMENT_POLY_4:
        return DataAugmentor::poly_4(x);
    case CTORCH_AUGMENT_POLY_5:
        return DataAugmentor::poly_5(x);
    case CTORCH_AUGMENT_RFF:
        if (d_features <= 0)
        {
            throw std::invalid_argument("d_features must be positive");
        }
        if (gamma <= 0.0)
        {
            throw std::invalid_argument("gamma must be positive");
        }
        return DataAugmentor::random_fourier_features(x, d_features, gamma);
    default:
        throw std::invalid_argument("invalid data augmentation type");
    }
}

} // namespace

static ml::NNOptimType as_nn_optim_type(CTorchNNOptimType optim_type)
{
    switch (optim_type)
    {
    case CTORCH_NN_OPTIM_SGD:
        return ml::NNOptimType::SGD;
    case CTORCH_NN_OPTIM_ADAGRAD:
        return ml::NNOptimType::ADAGRAD;
    case CTORCH_NN_OPTIM_RMSPROP:
        return ml::NNOptimType::RMSPROP;
    case CTORCH_NN_OPTIM_ADAM:
        return ml::NNOptimType::ADAM;
    case CTORCH_NN_OPTIM_ADAMW:
        return ml::NNOptimType::ADAMW;
    default:
        throw std::invalid_argument("invalid neural network optimizer type");
    }
}

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

struct CTorchSequential
{
    std::unique_ptr<ml::Sequential> value;
};

struct CTorchNNOptimizer
{
    std::unique_ptr<ml::NNOptimizer> value;
};

struct CTorchExpr
{
    std::shared_ptr<math::ASTNode> node;
};

struct CTorchLossFunction
{
    std::shared_ptr<LossFunction> impl;
};

struct CTorchParamMap
{
    std::unordered_map<std::string, double> values;
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

CTorchMatrix* make_matrix_handle(Matrix value)
{
    auto handle = std::make_unique<CTorchMatrix>();
    handle->value = std::move(value);
    return handle.release();
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

ml::Sequential& as_sequential_mut(CTorchSequential* handle)
{
    if (handle == nullptr || handle->value == nullptr)
    {
        throw std::invalid_argument("sequential model handle is null");
    }
    return *handle->value;
}

ml::NNOptimizer& as_nn_optimizer_mut(CTorchNNOptimizer* handle)
{
    if (handle == nullptr || handle->value == nullptr)
    {
        throw std::invalid_argument("neural network optimizer handle is null");
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
        return make_matrix_handle(as_matrix_ref(matrix).transpose());
    }, nullptr);
}

CTorchMatrix* ctorch_data_augment_no_op(const CTorchMatrix* x)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        return make_matrix_handle(DataAugmentor::no_op(as_matrix_ref(x)));
    }, nullptr);
}

CTorchMatrix* ctorch_data_augment_poly_2(const CTorchMatrix* x)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        return make_matrix_handle(DataAugmentor::poly_2(as_matrix_ref(x)));
    }, nullptr);
}

CTorchMatrix* ctorch_data_augment_poly_3(const CTorchMatrix* x)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        return make_matrix_handle(DataAugmentor::poly_3(as_matrix_ref(x)));
    }, nullptr);
}

CTorchMatrix* ctorch_data_augment_poly_4(const CTorchMatrix* x)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        return make_matrix_handle(DataAugmentor::poly_4(as_matrix_ref(x)));
    }, nullptr);
}

CTorchMatrix* ctorch_data_augment_poly_5(const CTorchMatrix* x)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        return make_matrix_handle(DataAugmentor::poly_5(as_matrix_ref(x)));
    }, nullptr);
}

CTorchMatrix* ctorch_data_augment_rff(const CTorchMatrix* x, int d_features, double gamma)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        return make_matrix_handle(apply_data_augmentation(as_matrix_ref(x), CTORCH_AUGMENT_RFF, d_features, gamma));
    }, nullptr);
}

CTorchMatrix* ctorch_data_augment_dispatch(
    const CTorchMatrix* x,
    CTorchDataAugmentationType augmentation,
    int d_features,
    double gamma)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        return make_matrix_handle(apply_data_augmentation(as_matrix_ref(x), augmentation, d_features, gamma));
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

CTorchMAB* ctorch_mab_create(int n_arms, float eps)
{
    return run_api<CTorchMAB*>([&]() -> CTorchMAB* {
        auto handle = std::make_unique<CTorchMAB>();
        handle->value = std::make_unique<ml::MAB>(n_arms, eps);
        return handle.release();
    }, nullptr);
}

void ctorch_mab_destroy(CTorchMAB* model)
{
    delete model;
}

bool ctorch_mab_select_arm(CTorchMAB* model, int* out_arm)
{
    return run_api<bool>([&]() -> bool {
        if (out_arm == nullptr)
        {
            throw std::invalid_argument("out_arm pointer is null");
        }
        *out_arm = as_mab_mut(model).select_arms();
        return true;
    }, false);
}

bool ctorch_mab_update(CTorchMAB* model, int arm, double reward)
{
    return run_api<bool>([&]() -> bool {
        as_mab_mut(model).update(arm, reward);
        return true;
    }, false);
}

bool ctorch_mab_set_epsilon(CTorchMAB* model, float eps)
{
    return run_api<bool>([&]() -> bool {
        as_mab_mut(model).set_epsilon(eps);
        return true;
    }, false);
}

CTorchUCB* ctorch_ucb_create(int n_arms)
{
    return run_api<CTorchUCB*>([&]() -> CTorchUCB* {
        auto handle = std::make_unique<CTorchUCB>();
        handle->value = std::make_unique<ml::UCB>(n_arms);
        return handle.release();
    }, nullptr);
}

void ctorch_ucb_destroy(CTorchUCB* model)
{
    delete model;
}

bool ctorch_ucb_select_arm(CTorchUCB* model, int* out_arm)
{
    return run_api<bool>([&]() -> bool {
        if (out_arm == nullptr)
        {
            throw std::invalid_argument("out_arm pointer is null");
        }
        *out_arm = as_ucb_mut(model).select_arms();
        return true;
    }, false);
}

bool ctorch_ucb_update(CTorchUCB* model, int arm, double reward)
{
    return run_api<bool>([&]() -> bool {
        as_ucb_mut(model).update(arm, reward);
        return true;
    }, false);
}

CTorchSequential* ctorch_sequential_create(void)
{
    return run_api<CTorchSequential*>([&]() -> CTorchSequential* {
        auto handle = std::make_unique<CTorchSequential>();
        handle->value = std::make_unique<ml::Sequential>();
        return handle.release();
    }, nullptr);
}

void ctorch_sequential_destroy(CTorchSequential* model)
{
    delete model;
}

bool ctorch_sequential_add_linear(CTorchSequential* model, int input_dim, int output_dim)
{
    return run_api<bool>([&]() -> bool {
        if (input_dim <= 0 || output_dim <= 0)
        {
            throw std::invalid_argument("linear layer dimensions must be positive");
        }
        as_sequential_mut(model).add_layer(std::make_shared<ml::LinearLayer>(input_dim, output_dim));
        return true;
    }, false);
}

bool ctorch_sequential_add_relu(CTorchSequential* model)
{
    return run_api<bool>([&]() -> bool {
        as_sequential_mut(model).add_layer(std::make_shared<ml::ReLULayer>());
        return true;
    }, false);
}

bool ctorch_sequential_add_sigmoid(CTorchSequential* model)
{
    return run_api<bool>([&]() -> bool {
        as_sequential_mut(model).add_layer(std::make_shared<ml::SigmoidLayer>());
        return true;
    }, false);
}

bool ctorch_sequential_add_tanh(CTorchSequential* model)
{
    return run_api<bool>([&]() -> bool {
        as_sequential_mut(model).add_layer(std::make_shared<ml::TanhLayer>());
        return true;
    }, false);
}

CTorchMatrix* ctorch_sequential_forward(CTorchSequential* model, const CTorchMatrix* x)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        Matrix output = as_sequential_mut(model).forward(as_matrix_ref(x));
        auto handle = std::make_unique<CTorchMatrix>();
        handle->value = std::move(output);
        return handle.release();
    }, nullptr);
}

bool ctorch_sequential_backward(CTorchSequential* model, const CTorchMatrix* dL_d_output)
{
    return run_api<bool>([&]() -> bool {
        as_sequential_mut(model).backward(as_matrix_ref(dL_d_output));
        return true;
    }, false);
}

bool ctorch_sequential_save(CTorchSequential* model, const char* filepath)
{
    return run_api<bool>([&]() -> bool {
        if (filepath == nullptr)
        {
            throw std::invalid_argument("filepath is null");
        }
        const bool ok = as_sequential_mut(model).save(filepath);
        if (!ok)
        {
            set_error("sequential save failed");
        }
        return ok;
    }, false);
}

bool ctorch_sequential_load(CTorchSequential* model, const char* filepath)
{
    return run_api<bool>([&]() -> bool {
        if (filepath == nullptr)
        {
            throw std::invalid_argument("filepath is null");
        }
        const bool ok = as_sequential_mut(model).load(filepath);
        if (!ok)
        {
            set_error("sequential load failed");
        }
        return ok;
    }, false);
}

CTorchNNOptimizer* ctorch_nn_optimizer_create(
    CTorchSequential* model,
    CTorchNNOptimType optim_type,
    float learning_rate,
    size_t batch_size,
    float beta1,
    float beta2,
    float epsilon,
    float rho,
    float weight_decay)
{
    return run_api<CTorchNNOptimizer*>([&]() -> CTorchNNOptimizer* {
        if (batch_size == 0)
        {
            throw std::invalid_argument("batch_size must be positive");
        }
        std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> params =
            as_sequential_mut(model).parameters();
        if (params.empty())
        {
            throw std::invalid_argument("sequential model has no linear parameters; add at least one linear layer");
        }
        auto handle = std::make_unique<CTorchNNOptimizer>();
        handle->value = std::make_unique<ml::NNOptimizer>(
            std::move(params),
            learning_rate,
            batch_size,
            as_nn_optim_type(optim_type),
            beta1,
            beta2,
            epsilon,
            rho,
            weight_decay);
        return handle.release();
    }, nullptr);
}

void ctorch_nn_optimizer_destroy(CTorchNNOptimizer* optimizer)
{
    delete optimizer;
}

bool ctorch_nn_optimizer_zero_grad(CTorchNNOptimizer* optimizer)
{
    return run_api<bool>([&]() -> bool {
        as_nn_optimizer_mut(optimizer).zero_grad();
        return true;
    }, false);
}

bool ctorch_nn_optimizer_step(CTorchNNOptimizer* optimizer)
{
    return run_api<bool>([&]() -> bool {
        as_nn_optimizer_mut(optimizer).step();
        return true;
    }, false);
}

CTorchMatrix* ctorch_matrix_eye(size_t dim)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        auto handle = std::make_unique<CTorchMatrix>();
        handle->value = Matrix::eye(dim);
        return handle.release();
    }, nullptr);
}

CTorchMatrix* ctorch_matrix_row(const CTorchMatrix* matrix, size_t row)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        const Matrix& m = as_matrix_ref(matrix);
        if (row >= m.numRows())
        {
            throw std::out_of_range("row index out of range");
        }
        Matrix out(1, m.numCols());
        for (size_t j = 0; j < m.numCols(); ++j)
        {
            out(0, j) = m(row, j);
        }
        auto handle = std::make_unique<CTorchMatrix>();
        handle->value = std::move(out);
        return handle.release();
    }, nullptr);
}

CTorchMatrix* ctorch_matrix_l2_norm_cols(const CTorchMatrix* matrix)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        auto handle = std::make_unique<CTorchMatrix>();
        handle->value = Matrix::l2_norm_cols(as_matrix_ref(matrix));
        return handle.release();
    }, nullptr);
}

CTorchMatrix* ctorch_matrix_center_cols(const CTorchMatrix* matrix)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        auto handle = std::make_unique<CTorchMatrix>();
        handle->value = Matrix::center_cols(as_matrix_ref(matrix));
        return handle.release();
    }, nullptr);
}

bool ctorch_matrix_euclidean_distance(const CTorchMatrix* lhs, const CTorchMatrix* rhs, double* out_value)
{
    return run_api<bool>([&]() -> bool {
        if (out_value == nullptr)
        {
            throw std::invalid_argument("out_value is null");
        }
        *out_value = as_matrix_ref(lhs).euclideanDistance(as_matrix_ref(rhs));
        return true;
    }, false);
}

bool ctorch_matrix_inner_product(const CTorchMatrix* lhs, const CTorchMatrix* rhs, double* out_value)
{
    return run_api<bool>([&]() -> bool {
        if (out_value == nullptr)
        {
            throw std::invalid_argument("out_value is null");
        }
        *out_value = as_matrix_ref(lhs).inner_product(as_matrix_ref(rhs));
        return true;
    }, false);
}

CTorchMatrix* ctorch_matrix_relu(const CTorchMatrix* matrix)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        auto handle = std::make_unique<CTorchMatrix>();
        handle->value = as_matrix_ref(matrix).relu();
        return handle.release();
    }, nullptr);
}

CTorchMatrix* ctorch_matrix_relu_deriv(const CTorchMatrix* matrix)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        auto handle = std::make_unique<CTorchMatrix>();
        handle->value = as_matrix_ref(matrix).relu_deriv();
        return handle.release();
    }, nullptr);
}

CTorchMatrix* ctorch_matrix_sigmoid(const CTorchMatrix* matrix)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        auto handle = std::make_unique<CTorchMatrix>();
        handle->value = as_matrix_ref(matrix).sigmoid();
        return handle.release();
    }, nullptr);
}

CTorchMatrix* ctorch_matrix_sigmoid_deriv(const CTorchMatrix* matrix)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        auto handle = std::make_unique<CTorchMatrix>();
        handle->value = as_matrix_ref(matrix).sigmoid_deriv();
        return handle.release();
    }, nullptr);
}

CTorchMatrix* ctorch_matrix_tanh(const CTorchMatrix* matrix)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        auto handle = std::make_unique<CTorchMatrix>();
        handle->value = as_matrix_ref(matrix).tanh();
        return handle.release();
    }, nullptr);
}

CTorchMatrix* ctorch_matrix_tanh_deriv(const CTorchMatrix* matrix)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        auto handle = std::make_unique<CTorchMatrix>();
        handle->value = as_matrix_ref(matrix).tanh_deriv();
        return handle.release();
    }, nullptr);
}

CTorchMatrix* ctorch_matrix_elm_wise_product(const CTorchMatrix* lhs, const CTorchMatrix* rhs)
{
    return run_api<CTorchMatrix*>([&]() -> CTorchMatrix* {
        auto handle = std::make_unique<CTorchMatrix>();
        handle->value = as_matrix_ref(lhs).elm_wise_product(as_matrix_ref(rhs));
        return handle.release();
    }, nullptr);
}

CTorchExpr* ctorch_expr_num(double value)
{
    return run_api<CTorchExpr*>([&]() -> CTorchExpr* {
        auto handle = std::make_unique<CTorchExpr>();
        handle->node = math::Num(value);
        return handle.release();
    }, nullptr);
}

CTorchExpr* ctorch_expr_var(const char* name)
{
    return run_api<CTorchExpr*>([&]() -> CTorchExpr* {
        if (name == nullptr)
        {
            throw std::invalid_argument("name is null");
        }
        auto handle = std::make_unique<CTorchExpr>();
        handle->node = math::Var(std::string(name));
        return handle.release();
    }, nullptr);
}

CTorchExpr* ctorch_expr_add(const CTorchExpr* lhs, const CTorchExpr* rhs)
{
    return run_api<CTorchExpr*>([&]() -> CTorchExpr* {
        if (lhs == nullptr || rhs == nullptr || !lhs->node || !rhs->node)
        {
            throw std::invalid_argument("expr operand is null");
        }
        auto handle = std::make_unique<CTorchExpr>();
        handle->node = lhs->node + rhs->node;
        return handle.release();
    }, nullptr);
}

CTorchExpr* ctorch_expr_sub(const CTorchExpr* lhs, const CTorchExpr* rhs)
{
    return run_api<CTorchExpr*>([&]() -> CTorchExpr* {
        if (lhs == nullptr || rhs == nullptr || !lhs->node || !rhs->node)
        {
            throw std::invalid_argument("expr operand is null");
        }
        auto handle = std::make_unique<CTorchExpr>();
        handle->node = lhs->node - rhs->node;
        return handle.release();
    }, nullptr);
}

CTorchExpr* ctorch_expr_mul(const CTorchExpr* lhs, const CTorchExpr* rhs)
{
    return run_api<CTorchExpr*>([&]() -> CTorchExpr* {
        if (lhs == nullptr || rhs == nullptr || !lhs->node || !rhs->node)
        {
            throw std::invalid_argument("expr operand is null");
        }
        auto handle = std::make_unique<CTorchExpr>();
        handle->node = lhs->node * rhs->node;
        return handle.release();
    }, nullptr);
}

CTorchExpr* ctorch_expr_div(const CTorchExpr* lhs, const CTorchExpr* rhs)
{
    return run_api<CTorchExpr*>([&]() -> CTorchExpr* {
        if (lhs == nullptr || rhs == nullptr || !lhs->node || !rhs->node)
        {
            throw std::invalid_argument("expr operand is null");
        }
        auto handle = std::make_unique<CTorchExpr>();
        handle->node = lhs->node / rhs->node;
        return handle.release();
    }, nullptr);
}

CTorchExpr* ctorch_expr_pow(const CTorchExpr* lhs, const CTorchExpr* rhs)
{
    return run_api<CTorchExpr*>([&]() -> CTorchExpr* {
        if (lhs == nullptr || rhs == nullptr || !lhs->node || !rhs->node)
        {
            throw std::invalid_argument("expr operand is null");
        }
        auto handle = std::make_unique<CTorchExpr>();
        handle->node = lhs->node ^ rhs->node;
        return handle.release();
    }, nullptr);
}

CTorchExpr* ctorch_expr_neg(const CTorchExpr* operand)
{
    return run_api<CTorchExpr*>([&]() -> CTorchExpr* {
        if (operand == nullptr || !operand->node)
        {
            throw std::invalid_argument("expr operand is null");
        }
        auto handle = std::make_unique<CTorchExpr>();
        handle->node = -operand->node;
        return handle.release();
    }, nullptr);
}

CTorchExpr* ctorch_expr_exp(const CTorchExpr* operand)
{
    return run_api<CTorchExpr*>([&]() -> CTorchExpr* {
        if (operand == nullptr || !operand->node)
        {
            throw std::invalid_argument("expr operand is null");
        }
        auto handle = std::make_unique<CTorchExpr>();
        handle->node = math::Exp(operand->node);
        return handle.release();
    }, nullptr);
}

CTorchExpr* ctorch_expr_log(const CTorchExpr* operand)
{
    return run_api<CTorchExpr*>([&]() -> CTorchExpr* {
        if (operand == nullptr || !operand->node)
        {
            throw std::invalid_argument("expr operand is null");
        }
        auto handle = std::make_unique<CTorchExpr>();
        handle->node = math::Log(operand->node);
        return handle.release();
    }, nullptr);
}

CTorchExpr* ctorch_expr_sqrt(const CTorchExpr* operand)
{
    return run_api<CTorchExpr*>([&]() -> CTorchExpr* {
        if (operand == nullptr || !operand->node)
        {
            throw std::invalid_argument("expr operand is null");
        }
        auto handle = std::make_unique<CTorchExpr>();
        handle->node = math::Sqrt(operand->node);
        return handle.release();
    }, nullptr);
}

CTorchExpr* ctorch_expr_abs(const CTorchExpr* operand)
{
    return run_api<CTorchExpr*>([&]() -> CTorchExpr* {
        if (operand == nullptr || !operand->node)
        {
            throw std::invalid_argument("expr operand is null");
        }
        auto handle = std::make_unique<CTorchExpr>();
        handle->node = math::Abs(operand->node);
        return handle.release();
    }, nullptr);
}

CTorchExpr* ctorch_expr_max(const CTorchExpr* lhs, const CTorchExpr* rhs)
{
    return run_api<CTorchExpr*>([&]() -> CTorchExpr* {
        if (lhs == nullptr || rhs == nullptr || !lhs->node || !rhs->node)
        {
            throw std::invalid_argument("expr operand is null");
        }
        auto handle = std::make_unique<CTorchExpr>();
        handle->node = math::Max(lhs->node, rhs->node);
        return handle.release();
    }, nullptr);
}

CTorchExpr* ctorch_expr_min(const CTorchExpr* lhs, const CTorchExpr* rhs)
{
    return run_api<CTorchExpr*>([&]() -> CTorchExpr* {
        if (lhs == nullptr || rhs == nullptr || !lhs->node || !rhs->node)
        {
            throw std::invalid_argument("expr operand is null");
        }
        auto handle = std::make_unique<CTorchExpr>();
        handle->node = math::Min(lhs->node, rhs->node);
        return handle.release();
    }, nullptr);
}

CTorchExpr* ctorch_expr_sigmoid(const CTorchExpr* operand)
{
    return run_api<CTorchExpr*>([&]() -> CTorchExpr* {
        if (operand == nullptr || !operand->node)
        {
            throw std::invalid_argument("expr operand is null");
        }
        auto handle = std::make_unique<CTorchExpr>();
        handle->node = math::Sigmoid(operand->node);
        return handle.release();
    }, nullptr);
}

void ctorch_expr_destroy(CTorchExpr* expr)
{
    delete expr;
}

const char* ctorch_expr_to_string(const CTorchExpr* expr)
{
    return run_api<const char*>(
        [&]() -> const char* {
            if (expr == nullptr || !expr->node)
            {
                throw std::invalid_argument("expr is null");
            }
            g_expr_to_string_buf = expr->node->to_string();
            return g_expr_to_string_buf.c_str();
        },
        "");
}

CTorchExpr* ctorch_expr_simplify(const CTorchExpr* expr)
{
    return run_api<CTorchExpr*>([&]() -> CTorchExpr* {
        if (expr == nullptr || !expr->node)
        {
            throw std::invalid_argument("expr is null");
        }
        auto handle = std::make_unique<CTorchExpr>();
        handle->node = expr->node->simplify();
        return handle.release();
    }, nullptr);
}

CTorchExpr* ctorch_expr_substitute(const CTorchExpr* expr, const char* var_name, const CTorchExpr* replacement)
{
    return run_api<CTorchExpr*>([&]() -> CTorchExpr* {
        if (expr == nullptr || !expr->node || var_name == nullptr || replacement == nullptr || !replacement->node)
        {
            throw std::invalid_argument("substitute arguments invalid");
        }
        auto handle = std::make_unique<CTorchExpr>();
        handle->node = expr->node->substitute(std::string(var_name), replacement->node);
        return handle.release();
    }, nullptr);
}

size_t ctorch_expr_variable_count(const CTorchExpr* expr)
{
    return run_api<size_t>(
        [&]() -> size_t {
            if (expr == nullptr || !expr->node)
            {
                throw std::invalid_argument("expr is null");
            }
            return expr->node->variables().size();
        },
        static_cast<size_t>(0));
}

bool ctorch_expr_variable_name(const CTorchExpr* expr, size_t index, char* out_buf, size_t out_buf_len)
{
    return run_api<bool>([&]() -> bool {
        if (expr == nullptr || !expr->node || out_buf == nullptr || out_buf_len == 0)
        {
            throw std::invalid_argument("ctorch_expr_variable_name invalid arguments");
        }
        const auto vars = expr->node->variables();
        if (index >= vars.size())
        {
            return false;
        }
        std::vector<std::string> sorted(vars.begin(), vars.end());
        std::sort(sorted.begin(), sorted.end());
        const std::string& name = sorted[index];
        if (name.size() + 1 > out_buf_len)
        {
            throw std::length_error("buffer too small for variable name");
        }
        std::memcpy(out_buf, name.c_str(), name.size() + 1);
        return true;
    }, false);
}

bool ctorch_expr_evaluate(
    const CTorchExpr* expr,
    size_t pair_count,
    const char** var_names,
    const double* var_values,
    double* out_value)
{
    return run_api<bool>([&]() -> bool {
        if (expr == nullptr || !expr->node || out_value == nullptr)
        {
            throw std::invalid_argument("ctorch_expr_evaluate invalid arguments");
        }
        if (pair_count > 0 && (var_names == nullptr || var_values == nullptr))
        {
            throw std::invalid_argument("var_names/var_values null with non-zero count");
        }
        std::shared_ptr<math::ASTNode> n = expr->node;
        for (size_t i = 0; i < pair_count; ++i)
        {
            if (var_names[i] == nullptr)
            {
                throw std::invalid_argument("null variable name");
            }
            n = n->substitute(std::string(var_names[i]), math::Num(var_values[i]));
        }
        n = n->simplify();
        const auto num = std::dynamic_pointer_cast<math::NumberNode>(n);
        if (num == nullptr)
        {
            throw std::runtime_error("expression did not evaluate to a constant (missing variable?)");
        }
        *out_value = num->getValue();
        return true;
    }, false);
}

bool ctorch_expr_gradient(
    const CTorchExpr* expr,
    size_t pair_count,
    const char** var_names,
    const double* var_values,
    double* out_partials)
{
    return run_api<bool>([&]() -> bool {
        if (expr == nullptr || !expr->node || out_partials == nullptr)
        {
            throw std::invalid_argument("ctorch_expr_gradient invalid arguments");
        }
        if (pair_count > 0 && (var_names == nullptr || var_values == nullptr))
        {
            throw std::invalid_argument("var_names/var_values null with non-zero count");
        }
        std::unordered_map<std::string, double> values;
        for (size_t i = 0; i < pair_count; ++i)
        {
            if (var_names[i] == nullptr)
            {
                throw std::invalid_argument("null variable name");
            }
            values[std::string(var_names[i])] = var_values[i];
        }
        math::Differentiator diff;
        const auto grads = diff.diff(expr->node, values);
        for (size_t i = 0; i < pair_count; ++i)
        {
            const std::string key(var_names[i]);
            out_partials[i] = grads.at(key);
        }
        return true;
    }, false);
}

CTorchLossFunction* ctorch_loss_regression_mse_create(int feature_dim, double l2_lambda)
{
    return run_api<CTorchLossFunction*>([&]() -> CTorchLossFunction* {
        if (feature_dim <= 0)
        {
            throw std::invalid_argument("feature_dim must be positive");
        }
        auto handle = std::make_unique<CTorchLossFunction>();
        handle->impl = std::make_shared<ml_bind_math::RegressionMSELoss>(static_cast<size_t>(feature_dim), l2_lambda);
        return handle.release();
    }, nullptr);
}

CTorchLossFunction* ctorch_loss_logistic_create(int feature_dim)
{
    return run_api<CTorchLossFunction*>([&]() -> CTorchLossFunction* {
        if (feature_dim <= 0)
        {
            throw std::invalid_argument("feature_dim must be positive");
        }
        (void)feature_dim;
        auto handle = std::make_unique<CTorchLossFunction>();
        handle->impl = std::make_shared<ml_bind_math::LogisticLossFn>();
        return handle.release();
    }, nullptr);
}

void ctorch_loss_destroy(CTorchLossFunction* loss)
{
    delete loss;
}

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
    double weight_decay)
{
    return run_api<CTorchParamMap*>([&]() -> CTorchParamMap* {
        if (loss == nullptr || !loss->impl)
        {
            throw std::invalid_argument("loss is null");
        }
        const Matrix& xTr = as_matrix_ref(x_train);
        const Matrix& yTr = as_matrix_ref(y_train);
        math::OptimParams params(
            as_optim_type(optim_type),
            learning_rate,
            max_iter,
            xTr,
            yTr,
            batch_size,
            beta1,
            beta2,
            epsilon,
            rho,
            weight_decay);
        math::Optimizer opt(params);
        std::unordered_map<std::string, double> theta0 =
            ml_bind_math::zero_initial_theta(*loss->impl, xTr, yTr);
        std::unordered_map<std::string, double> theta = opt.optimize(loss->impl, xTr, yTr, theta0);
        auto out = std::make_unique<CTorchParamMap>();
        out->values = std::move(theta);
        return out.release();
    }, nullptr);
}

void ctorch_param_map_destroy(CTorchParamMap* map)
{
    delete map;
}

size_t ctorch_param_map_size(const CTorchParamMap* map)
{
    return run_api<size_t>(
        [&]() -> size_t {
            if (map == nullptr)
            {
                throw std::invalid_argument("param map is null");
            }
            return map->values.size();
        },
        static_cast<size_t>(0));
}

bool ctorch_param_map_get(const CTorchParamMap* map, size_t index, char* key_buf, size_t key_buf_len, double* out_value)
{
    return run_api<bool>([&]() -> bool {
        if (map == nullptr || key_buf == nullptr || key_buf_len == 0 || out_value == nullptr)
        {
            throw std::invalid_argument("ctorch_param_map_get invalid arguments");
        }
        std::vector<std::pair<std::string, double>> sorted;
        sorted.reserve(map->values.size());
        for (const auto& p : map->values)
        {
            sorted.push_back(p);
        }
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        if (index >= sorted.size())
        {
            return false;
        }
        const std::string& key = sorted[index].first;
        if (key.size() + 1 > key_buf_len)
        {
            throw std::length_error("buffer too small for parameter name");
        }
        std::memcpy(key_buf, key.c_str(), key.size() + 1);
        *out_value = sorted[index].second;
        return true;
    }, false);
}

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
    int* iterations_out)
{
    return run_api<bool>([&]() -> bool {
        if (a == nullptr || b == nullptr || c == nullptr || solution_out == nullptr || objective_out == nullptr ||
            optimal_out == nullptr || unbounded_out == nullptr || iterations_out == nullptr)
        {
            throw std::invalid_argument("linear_program_solve null pointer argument");
        }
        const Matrix& A = as_matrix_ref(a);
        const size_t m = A.numRows();
        const size_t n = A.numCols();
        if (b_count != m || c_count != n)
        {
            throw std::invalid_argument("LP dimensions mismatch");
        }
        if (solution_capacity < n)
        {
            throw std::invalid_argument("solution_capacity too small");
        }
        std::vector<double> bvec(b, b + b_count);
        std::vector<double> cvec(c, c + c_count);
        math::LinearProgramSolver solver;
        const math::LinearProgramResult result = solver.solve(
            A,
            bvec,
            cvec,
            sense == CTORCH_LP_MINIMIZE ? math::LinearProgramSense::Minimize : math::LinearProgramSense::Maximize);
        for (size_t i = 0; i < n; ++i)
        {
            solution_out[i] = result.solution[i];
        }
        *objective_out = result.objective;
        *optimal_out = result.optimal;
        *unbounded_out = result.unbounded;
        *iterations_out = result.iterations;
        return true;
    }, false);
}

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
    int* iterations_out)
{
    return run_api<bool>([&]() -> bool {
        if (q == nullptr || c == nullptr || lower_bounds == nullptr || upper_bounds == nullptr || solution_out == nullptr ||
            objective_out == nullptr || converged_out == nullptr || iterations_out == nullptr)
        {
            throw std::invalid_argument("quadratic_program_solve null pointer argument");
        }
        const Matrix& Q = as_matrix_ref(q);
        if (Q.numRows() != n || Q.numCols() != n)
        {
            throw std::invalid_argument("Q must be n x n");
        }
        if (solution_capacity < n)
        {
            throw std::invalid_argument("solution_capacity too small");
        }
        std::vector<double> cvec(c, c + n);
        std::vector<double> lo(lower_bounds, lower_bounds + n);
        std::vector<double> hi(upper_bounds, upper_bounds + n);
        std::vector<double> eq;
        if (equality_len > 0)
        {
            if (equality_coeffs == nullptr)
            {
                throw std::invalid_argument("equality_coeffs null");
            }
            eq.assign(equality_coeffs, equality_coeffs + equality_len);
            if (eq.size() != n)
            {
                throw std::invalid_argument("equality coefficients length must equal n");
            }
        }
        std::vector<double> init;
        if (initial_len > 0)
        {
            if (initial_solution == nullptr || initial_len != n)
            {
                throw std::invalid_argument("initial_solution length must equal n");
            }
            init.assign(initial_solution, initial_solution + n);
        }
        math::QuadraticProgramSolver solver;
        const math::QuadraticProgramResult result =
            solver.solve(Q, cvec, lo, hi, eq, equality_value, init);
        for (size_t i = 0; i < n; ++i)
        {
            solution_out[i] = result.solution[i];
        }
        *objective_out = result.objective;
        *converged_out = result.converged;
        *iterations_out = result.iterations;
        return true;
    }, false);
}

} // extern "C"
