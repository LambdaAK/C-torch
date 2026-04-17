#include "bindings/ctorch_c_api.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "math/matrix.hpp"
#include "ml/knn.hpp"

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

} // namespace

struct CTorchMatrix
{
    Matrix value;
};

struct CTorchKNN
{
    std::unique_ptr<ml::KNN> value;
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
    if (handle == nullptr)
    {
        throw std::invalid_argument("knn handle is null");
    }
    return *handle->value;
}

ml::KNN& as_knn_mut(CTorchKNN* handle)
{
    if (handle == nullptr)
    {
        throw std::invalid_argument("knn handle is null");
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

} // extern "C"
