#include <gtest/gtest.h>
#include "math/matrix.hpp"
#include "ml/linearregression.hpp"

TEST(LinearRegression, ConstructorRejectsShapeMismatch) {
    Matrix xTr({{1.0, 2.0}, {3.0, 4.0}});
    Matrix yBad({{1.0}});
    EXPECT_THROW(ml::LinearRegression(xTr, yBad, 0.01, 10), std::invalid_argument);
}

TEST(LinearRegression, PredictRejectsFeatureMismatch) {
    Matrix xTr({{1.0, 2.0}, {2.0, 3.0}});
    Matrix yTr({{3.0, 5.0}});
    ml::LinearRegression model(xTr, yTr, 0.01, 20);

    Matrix bad_query({{10.0}});
    EXPECT_THROW(model.predict(bad_query), std::invalid_argument);
}

TEST(LinearRegression, ScoreRejectsSampleCountMismatch) {
    Matrix xTr({{1.0, 2.0}, {2.0, 3.0}});
    Matrix yTr({{3.0, 5.0}});
    ml::LinearRegression model(xTr, yTr, 0.01, 20);

    Matrix xTe({{1.0, 2.0}});
    Matrix yTe({{1.0, 0.0}});
    EXPECT_THROW(model.score(xTe, yTe), std::invalid_argument);
}
