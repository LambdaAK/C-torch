#include <gtest/gtest.h>
#include <cmath>
#include "math/matrix.hpp"

TEST(Matrix, DefaultConstructorIsOneByOneZero) {
    Matrix m;
    EXPECT_EQ(m.numRows(), 1u);
    EXPECT_EQ(m.numCols(), 1u);
    EXPECT_DOUBLE_EQ(m(0, 0), 0.0);
}

TEST(Matrix, InnerProductRowVectors) {
    Matrix a({{1.0, 2.0, 3.0}});
    Matrix b({{4.0, 5.0, 6.0}});
    EXPECT_DOUBLE_EQ(a.inner_product(b), 32.0);
}

TEST(Matrix, Transpose) {
    Matrix m({{1.0, 2.0}, {3.0, 4.0}});
    Matrix t = m.transpose();
    EXPECT_DOUBLE_EQ(t(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(t(0, 1), 3.0);
    EXPECT_DOUBLE_EQ(t(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(t(1, 1), 4.0);
}

TEST(Matrix, CopyAssignment) {
    Matrix a({{1.0, 2.0}});
    Matrix b;
    b = a;
    EXPECT_DOUBLE_EQ(b(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(b(0, 1), 2.0);
    b(0, 0) = 9.0;
    EXPECT_DOUBLE_EQ(a(0, 0), 1.0);
}

TEST(Matrix, EmptyInitializerListCreatesZeroByZero) {
    Matrix m({});
    EXPECT_EQ(m.numRows(), 0u);
    EXPECT_EQ(m.numCols(), 0u);
}

TEST(Matrix, LargeMatrixMultiplicationProducesExpectedValues) {
    Matrix a(64, 64);
    Matrix b(64, 64);

    for (size_t i = 0; i < 64; ++i) {
        for (size_t j = 0; j < 64; ++j) {
            a(i, j) = 1.0;
            b(i, j) = 2.0;
        }
    }

    Matrix result = a * b;

    EXPECT_EQ(result.numRows(), 64u);
    EXPECT_EQ(result.numCols(), 64u);
    for (size_t i = 0; i < 64; ++i) {
        for (size_t j = 0; j < 64; ++j) {
            EXPECT_DOUBLE_EQ(result(i, j), 128.0);
        }
    }
}

TEST(Matrix, LargeVectorInnerProductAndDistance) {
    Matrix a(1, 8192);
    Matrix b(1, 8192);

    for (size_t i = 0; i < 8192; ++i) {
        a(0, i) = 1.0;
        b(0, i) = 2.0;
    }

    EXPECT_DOUBLE_EQ(a.inner_product(b), 16384.0);
    EXPECT_NEAR(a.euclideanDistance(b), std::sqrt(8192.0), 1e-9);
}
