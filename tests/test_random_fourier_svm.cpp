#include <gtest/gtest.h>
#include "math/matrix.hpp"
#include "ml/randomfouriersvm.hpp"

TEST(RandomFourierSVM, ConstructorRejectsInvalidParams) {
    Matrix xTr({{0.0}, {1.0}});
    Matrix yTr({{-1.0, 1.0}});
    EXPECT_THROW(ml::RandomFourierSVM(xTr, yTr, 0, 0.5, 0.01, 10, 1.0), std::invalid_argument);
    EXPECT_THROW(ml::RandomFourierSVM(xTr, yTr, -1, 0.5, 0.01, 10, 1.0), std::invalid_argument);
    EXPECT_THROW(ml::RandomFourierSVM(xTr, yTr, 8, 0.0, 0.01, 10, 1.0), std::invalid_argument);
}

TEST(RandomFourierSVM, ScoreRejectsShapeMismatch) {
    Matrix xTr({{0.0}, {1.0}, {2.0}});
    Matrix yTr({{-1.0, -1.0, 1.0}});
    ml::RandomFourierSVM model(xTr, yTr, 8, 0.5, 0.01, 50, 1.0);

    Matrix xTe({{0.0}, {1.0}});
    Matrix yTe_bad_rows({{1.0}, {0.0}});
    Matrix yTe_bad_len({{-1.0}});
    EXPECT_THROW(model.score(xTe, yTe_bad_rows), std::invalid_argument);
    EXPECT_THROW(model.score(xTe, yTe_bad_len), std::invalid_argument);
}
