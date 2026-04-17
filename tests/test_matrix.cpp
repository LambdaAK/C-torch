#include <gtest/gtest.h>
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
