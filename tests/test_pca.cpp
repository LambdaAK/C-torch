#include <gtest/gtest.h>
#include <stdexcept>
#include "math/matrix.hpp"
#include "ml/pca.hpp"

TEST(PCA, ProjectionMatrixShape) {
    Matrix X(20, 3);
    for (size_t i = 0; i < X.numRows(); ++i) {
        for (size_t j = 0; j < X.numCols(); ++j) {
            X(i, j) = static_cast<double>((i + 1) * (j + 2) % 7) - 2.0;
        }
    }
    Matrix col_mean(1, X.numCols());
    for (size_t j = 0; j < X.numCols(); ++j) {
        double s = 0.0;
        for (size_t i = 0; i < X.numRows(); ++i) {
            s += X(i, j);
        }
        col_mean(0, j) = s / static_cast<double>(X.numRows());
    }
    Matrix Xc(X.numRows(), X.numCols());
    for (size_t i = 0; i < X.numRows(); ++i) {
        for (size_t j = 0; j < X.numCols(); ++j) {
            Xc(i, j) = X(i, j) - col_mean(0, j);
        }
    }

    ml::PCA pca(Xc);
    Matrix P = pca.compute_projection_mat(2, 200, 1e-6f);
    EXPECT_EQ(P.numRows(), X.numCols());
    EXPECT_EQ(P.numCols(), 2u);
}

TEST(PCA, InvalidKThrows) {
    Matrix X(20, 3);
    for (size_t i = 0; i < X.numRows(); ++i) {
        for (size_t j = 0; j < X.numCols(); ++j) {
            X(i, j) = static_cast<double>((i + 1) * (j + 2) % 7) - 2.0;
        }
    }
    Matrix col_mean(1, X.numCols());
    for (size_t j = 0; j < X.numCols(); ++j) {
        double s = 0.0;
        for (size_t i = 0; i < X.numRows(); ++i) {
            s += X(i, j);
        }
        col_mean(0, j) = s / static_cast<double>(X.numRows());
    }
    Matrix Xc(X.numRows(), X.numCols());
    for (size_t i = 0; i < X.numRows(); ++i) {
        for (size_t j = 0; j < X.numCols(); ++j) {
            Xc(i, j) = X(i, j) - col_mean(0, j);
        }
    }
    ml::PCA pca(Xc);
    EXPECT_THROW(pca.compute_projection_mat(0, 50, 1e-6f), std::invalid_argument);
    EXPECT_THROW(pca.compute_projection_mat(4, 50, 1e-6f), std::invalid_argument);
}
