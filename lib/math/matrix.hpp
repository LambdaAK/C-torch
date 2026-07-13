#pragma once

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

/**
 * @brief Dense matrix utility class used across models and optimizers.
 *
 * Storage is one contiguous, row-major `double` buffer aligned to 64 bytes (a
 * cache line, and wide enough for any SIMD width). The class owns its buffer
 * with value semantics: copies are deep, and moves are O(1) pointer steals, so
 * returning or reassigning a temporary never copies its data.
 *
 * Results of arithmetic are allocated without zero-filling, since every element
 * is overwritten. Large buffers are recycled through a caching allocator, so
 * same-shaped temporaries in a training loop do not page-fault fresh memory. Matmul uses a packed, cache-blocked, register-tiled kernel, and
 * large ops run on a persistent thread pool (see `parallel.hpp`).
 */
class Matrix
{
private:
    size_t rows, cols;      ///< Matrix dimensions.
    double *data_ = nullptr; ///< Row-major element storage (`rows * cols`, 64-byte aligned), or null when empty.

    struct Uninitialized
    {
    };

    /** @brief Allocates a `rows x cols` buffer without initializing it. */
    Matrix(size_t rows, size_t cols, Uninitialized);

    static double *allocate(size_t count);
    static void deallocate(double *ptr, size_t count) noexcept;

public:
    /** @brief Memory alignment of the element buffer, in bytes. */
    static constexpr size_t kAlignment = 64;

    /**
     * @brief Constructs a `1 x 1` matrix holding zero.
     */
    Matrix();

    /**
     * @brief Constructs a zero-initialized matrix of fixed shape.
     * @param rows Number of rows.
     * @param cols Number of columns.
     */
    Matrix(size_t rows, size_t cols);

    /**
     * @brief Constructs a matrix from nested initializer lists.
     * @param init_list Row-wise literal values.
     */
    Matrix(std::initializer_list<std::initializer_list<double>> init_list);

    /** @brief Deep copy. */
    Matrix(const Matrix &other);

    /** @brief Steals `other`'s buffer; `other` is left as an empty `0 x 0` matrix. */
    Matrix(Matrix &&other) noexcept;

    /** @brief Deep copy assignment. Reuses the existing buffer when the sizes match. */
    Matrix &operator=(const Matrix &other);

    /** @brief Move assignment; `other` is left as an empty `0 x 0` matrix. */
    Matrix &operator=(Matrix &&other) noexcept;

    ~Matrix();

    /**
     * @brief Creates an identity matrix.
     * @param dim Matrix dimension.
     * @return `dim x dim` identity matrix.
     */
    static Matrix eye(size_t dim);

    /**
     * @brief Computes column-wise L2 norms.
     * @param X Input matrix.
     * @return Matrix with each column divided by its L2 norm.
     */
    static Matrix l2_norm_cols(const Matrix& X);

    /**
     * @brief Centers each column by subtracting its mean.
     * @param X Input matrix.
     * @return Column-centered matrix with same shape as `X`.
     */
    static Matrix center_cols(const Matrix& X);

    /** @brief Returns row count. */
    size_t numRows() const noexcept { return rows; }

    /** @brief Returns column count. */
    size_t numCols() const noexcept { return cols; }

    /** @brief Returns the number of elements (`numRows() * numCols()`). */
    size_t size() const noexcept { return rows * cols; }

    /** @brief Raw row-major element buffer (null when empty). Element `(i, j)` is `data()[i * numCols() + j]`. */
    double *data() noexcept { return data_; }

    /** @brief Raw row-major element buffer (null when empty). */
    const double *data() const noexcept { return data_; }

    /**
     * @brief Bounds-checked read accessor.
     * @param row Row index.
     * @param col Column index.
     * @return Element value at `(row, col)`.
     */
    double at(size_t row, size_t col) const
    {
        if (row >= rows || col >= cols)
            throw std::out_of_range("Matrix::at(): Index out of bounds");
        return data_[row * cols + col];
    }

    /**
     * @brief Bounds-checked mutable element accessor.
     * @param row Row index.
     * @param col Column index.
     * @return Reference to element at `(row, col)`.
     */
    double &operator()(size_t row, size_t col)
    {
        if (row >= rows || col >= cols)
            throw std::out_of_range("Matrix::operator(): Index out of bounds");
        return data_[row * cols + col];
    }

    /**
     * @brief Bounds-checked const element accessor.
     * @param row Row index.
     * @param col Column index.
     * @return Element value at `(row, col)`.
     */
    double operator()(size_t row, size_t col) const
    {
        if (row >= rows || col >= cols)
            throw std::out_of_range("Matrix::operator() const: Index out of bounds");
        return data_[row * cols + col];
    }

    /**
     * @brief Prints matrix contents to standard output.
     */
    void print() const;

    /**
     * @brief Returns transposed matrix.
     * @return Matrix with rows and columns swapped.
     */
    Matrix transpose() const;

    /**
     * @brief Element-wise matrix addition.
     * @param other Right-hand matrix operand.
     * @return Sum matrix.
     */
    Matrix operator+(const Matrix &other) const;

    /**
     * @brief Element-wise matrix subtraction.
     * @param other Right-hand matrix operand.
     * @return Difference matrix.
     */
    Matrix operator-(const Matrix &other) const;

    /**
     * @brief Scalar multiplication.
     * @param scalar Scalar factor.
     * @return Scaled matrix.
     */
    Matrix operator*(double scalar) const;

    /**
     * @brief Matrix multiplication.
     * @param other Right-hand matrix operand.
     * @return Product matrix.
     */
    Matrix operator*(const Matrix &other) const;

    /**
     * @brief Element-wise equality check.
     * @param other Matrix to compare.
     * @return `true` if shapes and all elements match.
     */
    bool operator==(const Matrix &other) const;

    /**
     * @brief Computes Euclidean distance between row vectors.
     * @param other Matrix to compare against.
     * @return Euclidean distance.
     */
    double euclideanDistance(const Matrix &other) const;

    /**
     * @brief Returns each row as an independent `1 x numCols` matrix.
     * @return Vector of row matrices.
     */
    std::vector<Matrix> rowsAsMatrices() const;

    /**
     * @brief Computes inner product between compatible row vectors.
     * @param other Matrix to dot with.
     * @return Dot product value.
     */
    double inner_product(const Matrix &other) const;

    /**
     * @brief Applies element-wise ReLU.
     * @return Matrix with `max(0, x)` applied.
     */
    Matrix relu() const;

    /**
     * @brief Computes derivative mask of ReLU.
     * @return Matrix with `1` where input > 0, else `0`.
     */
    Matrix relu_deriv() const;

    /**
     * @brief Applies element-wise sigmoid.
     * @return Sigmoid-transformed matrix.
     */
    Matrix sigmoid() const;

    /**
     * @brief Computes derivative of sigmoid element-wise.
     * @return Matrix of sigmoid derivatives.
     */
    Matrix sigmoid_deriv() const;

    /**
     * @brief Applies element-wise hyperbolic tangent.
     * @return Tanh-transformed matrix.
     */
    Matrix tanh() const;

    /**
     * @brief Computes derivative of tanh element-wise.
     * @return Matrix of tanh derivatives.
     */
    Matrix tanh_deriv() const;

    /**
     * @brief Performs element-wise product with another matrix.
     * @param other Matrix operand with matching shape.
     * @return Element-wise product matrix.
     */
    Matrix elm_wise_product(const Matrix &other) const;
};

/**
 * @brief Scalar-matrix multiplication with scalar on the left.
 * @param scalar Scalar factor.
 * @param m Matrix operand.
 * @return Scaled matrix.
 */
Matrix operator*(double scalar, const Matrix &m);

#endif // MATRIX_HPP
