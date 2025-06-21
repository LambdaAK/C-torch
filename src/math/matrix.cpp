#include "matrix.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

// Constructor: Zero-initialized matrix
Matrix::Matrix(size_t rows, size_t cols)
    : rows(rows), cols(cols), data(rows * cols, 0.0) {}

// Constructor: Uses double initializer list to construct the matrix
Matrix::Matrix(std::initializer_list<std::initializer_list<double>> init_list) : rows(init_list.size()), cols(init_list.begin()->size()), data(rows * cols)
{
    size_t i = 0;
    // Validate all rows are the same size
    for (const auto &row : init_list)
    {
        if (row.size() != cols)
        {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
        std::copy(row.begin(), row.end(), data.begin() + (i * cols));
        ++i;
    }
}

// Default constructor: Creates a 1x1 matrix filled with zero
Matrix::Matrix()
{
    data.resize(1, 0.0);
}

// Copy constructor
Matrix::Matrix(const Matrix &other) = default;

Matrix Matrix::eye(size_t dim)
{
    Matrix result(dim, dim);
    // Make each entry on the diagonal 1
    for (size_t i = 0; i < dim; ++i)
    {
        result(i, i) = 1.0;
    }
    return result;
}

// Accessors below
size_t Matrix::numRows() const
{
    return rows;
}

size_t Matrix::numCols() const
{
    return cols;
}

double Matrix::at(size_t row, size_t col) const
{
    if (row >= rows || col >= cols)
        throw std::out_of_range("Matrix::at(): Index out of bounds");
    return data[row * cols + col];
}

double &Matrix::operator()(size_t row, size_t col)
{
    if (row >= rows || col >= cols)
        throw std::out_of_range("Matrix::operator(): Index out of bounds");
    return data[row * cols + col];
}

double Matrix::operator()(size_t row, size_t col) const
{
    if (row >= rows || col >= cols)
        throw std::out_of_range("Matrix::operator() const: Index out of bounds");
    return data[row * cols + col];
}

void Matrix::print() const
{
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
            std::cout << (*this)(i, j) << " ";
        std::cout << "\n";
    }
}

Matrix Matrix::transpose() const
{
    Matrix result(cols, rows); // make a matrix with the swapped dimensions

    // Map (i, j) -> (j, i)
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            result(j, i) = (*this)(i, j);
    return result;
}

Matrix Matrix::operator+(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix dimensions must match for addition");
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows * cols; ++i)
        result.data[i] = data[i] + other.data[i];
    return result;
}

Matrix Matrix::operator-(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix dimensions must match for addition");
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows * cols; ++i)
        result.data[i] = data[i] - other.data[i];
    return result;
}

// scalar multiplication
Matrix Matrix::operator*(double scalar) const
{
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows * cols; ++i)
        result.data[i] = data[i] * scalar;
    return result;
}

// opposite-direction scalar multiplication
Matrix operator*(double scalar, const Matrix &m)
{
    Matrix result(m.numRows(), m.numCols());

    for (size_t i = 0; i < m.numRows(); ++i)
        for (size_t j = 0; j < m.numCols(); ++j)
            result(i, j) = scalar * m(i, j);
    return result;
}

// matrix multiplication
Matrix Matrix::operator*(const Matrix &m) const
{
    if (cols != m.rows)
        throw std::invalid_argument("Matrix dimensions are incompatible for multiplication: (" + 
            std::to_string(rows) + "x" + std::to_string(cols) + ") * (" + 
            std::to_string(m.rows) + "x" + std::to_string(m.cols) + ")");

    Matrix result(rows, m.cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < m.cols; ++j)
            for (size_t k = 0; k < cols; ++k)
                result(i, j) += (*this)(i, k) * m(k, j);
    return result;
}

// equality check with tolerance
bool Matrix::operator==(const Matrix &other) const
{
    const double EPSILON = 1e-10;
    if (rows != other.rows || cols != other.cols)
        return false;

    for (size_t i = 0; i < rows * cols; ++i)
        if (std::abs(data[i] - other.data[i]) > EPSILON)
            return false;
    return true;
}

double Matrix::euclideanDistance(const Matrix &other) const
{
    bool this_is_vector = (rows == 1 || cols == 1);
    bool other_is_vector = (other.rows == 1 || other.cols == 1);

    if (!this_is_vector || !other_is_vector)
        throw std::invalid_argument("Both inputs must be vectors for Euclidean distance");

    size_t this_length = rows * cols;
    size_t other_length = other.rows * other.cols;

    if (this_length != other_length)
        throw std::invalid_argument("Vectors must have the same length for Euclidean distance");

    // Euclidean distance formula: sqrt(sum((a_i - b_i)^2))
    double sum_squared_diff = 0.0;
    for (size_t i = 0; i < this_length; ++i)
    {
        double diff = data[i] - other.data[i];
        sum_squared_diff += diff * diff;
    }
    return std::sqrt(sum_squared_diff);
}

// Return the rows of a matrix as matrix objects. Expensive, so call sparingly
std::vector<Matrix> Matrix::rowsAsMatrices() const
{
    std::vector<Matrix> matrices;

    for (size_t i = 0; i < rows; ++i)
    {
        Matrix row(1, cols); // one row, same number of columns
        for (size_t j = 0; j < cols; ++j)
        {
            row(0, j) = (*this)(i, j);
        }
        matrices.push_back(row);
    }
    return matrices;
}

double Matrix::inner_product(const Matrix &other) const
{
    /*
        One of the following conditions must hold:
        1. Both matrices are row vectors of the same length
        2. Both matrices are column vectors of the same length
    */
    if (rows != other.rows || cols != other.cols || !(rows == 1 || cols == 1))
        throw std::invalid_argument("Dimension mismatch.");
    double result = 0.0;
    // element-wise product
    for (size_t i = 0; i < rows; ++i)
        result += data[i] * other.data[i];

    return result;
}

Matrix Matrix::relu() const
{
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            result(i, j) = std::max(0.0, data[i * cols + j]);
        }
    }
    return result;
}

Matrix Matrix::relu_deriv() const
{
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            if (data[i * cols + j] <= 0)
            {
                result(i, j) = 0;
            }
            else
            {
                result(i, j) = 1.0;
            }
        }
    }
    return result;
}

Matrix Matrix::sigmoid() const
{
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            result(i, j) = 1.0 / (1.0 + std::exp(-data[i * cols + j]));
        }
    }
    return result;
}

Matrix Matrix::sigmoid_deriv() const
{
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            double sigmoid = 1.0 / (1.0 + std::exp(-data[i * cols + j]));
            result(i, j) = sigmoid * (1 - sigmoid);
        }
    }
    return result;
}

Matrix Matrix::tanh() const
{
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            result(i, j) = std::tanh(data[i * cols + j]);
        }
    }
    return result;
}

Matrix Matrix::tanh_deriv() const
{
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            double tanh = std::tanh(data[i * cols + j]);
            result(i, j) = 1 - tanh * tanh;
        }
    }
    return result;
}

Matrix Matrix::elm_wise_product(const Matrix &other) const
{
    // first, check that the dimensions match
    if (rows != other.rows || cols != other.cols)
    {
        throw std::invalid_argument("Matrix dimensions must match for element-wise product");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            result(i, j) = data[i * cols + j] * other.data[i * cols + j];
        }
    }
    return result;
}

Matrix Matrix::l2_norm_cols(const Matrix &X)
{
    std::vector<float> l2_norms(X.numCols(), 0);

    for (size_t i = 0; i < X.numRows(); ++i)
    {
        for (size_t j = 0; j < X.numCols(); ++j)
        {
            l2_norms[j] += X(i, j) * X(i, j);
        }
    }

    Matrix Norm_X = X;
    for (size_t i = 0; i < X.numRows(); ++i)
    {
        for (size_t j = 0; j < X.numCols(); ++j)
        {
            Norm_X(i, j) = Norm_X(i, j) / std::sqrt(l2_norms[j]);
        }
    }

    return Norm_X;
}

Matrix Matrix::center_cols(const Matrix &X)
{
    std::vector<float> avgs(X.numCols(), 0);

    for (size_t i = 0; i < X.numRows(); ++i)
    {
        for (size_t j = 0; j < X.numCols(); ++j)
        {
            avgs[j] += X(i, j);
        }
    }

    Matrix Centered_X = X;
    for (size_t i = 0; i < X.numRows(); ++i)
    {
        for (size_t j = 0; j < X.numCols(); ++j)
        {
            Centered_X(i, j) = Centered_X(i, j) - (avgs[j] / X.numRows());
        }
    }

    return Centered_X;
}
