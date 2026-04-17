#include <gtest/gtest.h>
#include "math/matrix.hpp"
#include "ml/gaussian_nb.hpp"

TEST(GaussianNB, SeparableTwoClass) {
    Matrix xTr(6, 2);
    Matrix yTr(1, 6);
    for (int i = 0; i < 3; ++i) {
        xTr(i, 0) = 0.1 * static_cast<double>(i);
        xTr(i, 1) = 0.0;
        yTr(0, static_cast<size_t>(i)) = 0;
    }
    for (int i = 0; i < 3; ++i) {
        xTr(3 + i, 0) = 5.0 + 0.1 * static_cast<double>(i);
        xTr(3 + i, 1) = 5.0;
        yTr(0, static_cast<size_t>(3 + i)) = 1;
    }

    ml::GaussianNB model(xTr, yTr);

    Matrix q0(1, 2);
    q0(0, 0) = 0.0;
    q0(0, 1) = 0.0;
    EXPECT_EQ(model.predict(q0), 0);

    Matrix q1(1, 2);
    q1(0, 0) = 5.0;
    q1(0, 1) = 5.0;
    EXPECT_EQ(model.predict(q1), 1);

    Matrix xTe(2, 2);
    xTe(0, 0) = 0.05;
    xTe(0, 1) = 0.0;
    xTe(1, 0) = 5.05;
    xTe(1, 1) = 5.0;
    Matrix yTe(1, 2);
    yTe(0, 0) = 0;
    yTe(0, 1) = 1;
    EXPECT_DOUBLE_EQ(model.score(xTe, yTe), 1.0);
}
