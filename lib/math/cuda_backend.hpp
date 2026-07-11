#pragma once

/**
 * @file cuda_backend.hpp
 * @brief Optional CUDA acceleration for dense `Matrix` kernels.
 *
 * Matrices stay in host memory. Each accelerated op copies its inputs to the
 * device, runs, and copies the result back, so callers never see a device
 * pointer. `Matrix` routes to these functions only when `should_use_*()`
 * returns true; otherwise it uses its multithreaded CPU path.
 *
 * Builds without CUDA (`CTORCH_ENABLE_CUDA=OFF`, the default) link a stub where
 * `is_available()` is always false.
 */

#include <cstddef>

namespace ctorch::cuda
{
/** @brief Elementwise ops on one input. `Scale` multiplies by `scalar`. */
enum class UnaryOp
{
    Scale,
    Relu,
    ReluDeriv,
    Sigmoid,
    SigmoidDeriv,
    Tanh,
    TanhDeriv,
};

/** @brief Elementwise ops on two inputs of equal length. */
enum class BinaryOp
{
    Add,
    Sub,
    Mul,
};

/** @brief True when the library was built with CUDA support. */
bool compiled_with_cuda();

/** @brief True when built with CUDA and at least one CUDA device is visible. */
bool is_available();

/**
 * @brief Enables or disables GPU dispatch at runtime.
 *
 * Defaults to enabled, unless the environment variable `CTORCH_DEVICE=cpu`
 * is set at startup.
 */
void set_enabled(bool enabled);

/** @brief Whether GPU dispatch is currently enabled. */
bool is_enabled();

/**
 * @brief Sets the minimum problem sizes that are sent to the GPU.
 * @param matmul_min_work Minimum `m * k * n` for matrix multiplication.
 * @param elementwise_min_elems Minimum element count for elementwise ops and transpose.
 *
 * Smaller problems stay on the CPU because host/device copies would dominate.
 * Tests set both to 0 to force the GPU path.
 */
void set_thresholds(size_t matmul_min_work, size_t elementwise_min_elems);

/** @brief Whether an `(m x k) * (k x n)` product should run on the GPU. */
bool should_use_matmul(size_t m, size_t k, size_t n);

/** @brief Whether an elementwise op or transpose over `n` elements should run on the GPU. */
bool should_use_elementwise(size_t n);

/** @brief Row-major `c(m x n) = a(m x k) * b(k x n)` via cuBLAS. */
void matmul(const double *a, const double *b, double *c, size_t m, size_t k, size_t n);

/** @brief `out[i] = op(in[i])` for `n` elements. */
void unary(UnaryOp op, const double *in, double *out, size_t n, double scalar = 0.0);

/** @brief `out[i] = op(a[i], b[i])` for `n` elements. */
void binary(BinaryOp op, const double *a, const double *b, double *out, size_t n);

/** @brief Row-major transpose of a `rows x cols` matrix into `out` (`cols x rows`). */
void transpose(const double *in, double *out, size_t rows, size_t cols);

namespace detail
{
/** @brief Device probe implemented by the CUDA or stub translation unit. */
bool device_present();
} // namespace detail
} // namespace ctorch::cuda
