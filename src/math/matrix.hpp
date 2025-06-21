#pragma once

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <iostream>
#include <initializer_list>
#include <stdexcept>

class Matrix
{
private:
    size_t rows, cols;
    std::vector<double> data;

public:
    // Constructors
    Matrix();
    Matrix(size_t rows, size_t cols);
    Matrix(std::initializer_list<std::initializer_list<double>> init_list);
    Matrix(const Matrix &other);

    // Static methods
    static Matrix eye(size_t dim);
    static Matrix l2_norm_cols(const Matrix& X);
    static Matrix center_cols(const Matrix& X);

    // Accessors for dimensions
    size_t numRows() const;
    size_t numCols() const;

    // Element access methods
    double at(size_t row, size_t col) const;
    double &operator()(size_t row, size_t col);
    double operator()(size_t row, size_t col) const;

    // Display method
    void print() const;

    // Matrix operations
    Matrix transpose() const;

    // Arithmetic operators
    Matrix operator+(const Matrix &other) const;
    Matrix operator-(const Matrix &other) const;
    Matrix operator*(double scalar) const;
    Matrix operator*(const Matrix &other) const;

    // Equality operator
    bool operator==(const Matrix &other) const;

    // Euclidean distance
    double euclideanDistance(const Matrix &other) const;

    std::vector<Matrix> rowsAsMatrices() const;

    double inner_product(const Matrix &other) const;

    Matrix relu() const;
    Matrix relu_deriv() const;
    Matrix sigmoid() const;
    Matrix sigmoid_deriv() const;
    Matrix tanh() const;
    Matrix tanh_deriv() const;

    Matrix elm_wise_product(const Matrix &other) const;
};

// Global operators
Matrix operator*(double scalar, const Matrix &m);

#endif // MATRIX_HPP
