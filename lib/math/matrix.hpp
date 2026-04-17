#pragma once

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <iostream>
#include <initializer_list>
#include <stdexcept>

/**
 * @brief Dense matrix utility class used across models and optimizers.
 *
 * Storage is row-major in a contiguous `std::vector<double>`.
 */
class Matrix
{
private:
    size_t rows, cols;        ///< Matrix dimensions.
    std::vector<double> data; ///< Row-major element storage.

public:
    /**
     * @brief Constructs an empty matrix (`0 x 0`).
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

    /**
     * @brief Copy constructor.
     * @param other Matrix to copy.
     */
    Matrix(const Matrix &other);

    /**
     * @brief Copy assignment.
     * @param other Matrix to copy from.
     * @return Reference to assigned matrix.
     */
    Matrix &operator=(const Matrix &other);

    /**
     * @brief Creates an identity matrix.
     * @param dim Matrix dimension.
     * @return `dim x dim` identity matrix.
     */
    static Matrix eye(size_t dim);

    /**
     * @brief Computes column-wise L2 norms.
     * @param X Input matrix.
     * @return Row matrix containing one norm per column.
     */
    static Matrix l2_norm_cols(const Matrix& X);

    /**
     * @brief Centers each column by subtracting its mean.
     * @param X Input matrix.
     * @return Column-centered matrix with same shape as `X`.
     */
    static Matrix center_cols(const Matrix& X);

    /**
     * @brief Returns row count.
     * @return Number of rows.
     */
    size_t numRows() const;

    /**
     * @brief Returns column count.
     * @return Number of columns.
     */
    size_t numCols() const;

    /**
     * @brief Bounds-checked read accessor.
     * @param row Row index.
     * @param col Column index.
     * @return Element value at `(row, col)`.
     */
    double at(size_t row, size_t col) const;

    /**
     * @brief Mutable element accessor.
     * @param row Row index.
     * @param col Column index.
     * @return Reference to element at `(row, col)`.
     */
    double &operator()(size_t row, size_t col);

    /**
     * @brief Const element accessor.
     * @param row Row index.
     * @param col Column index.
     * @return Element value at `(row, col)`.
     */
    double operator()(size_t row, size_t col) const;

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
