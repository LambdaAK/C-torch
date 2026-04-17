#include <gtest/gtest.h>

#include "math/dataaugmentor.hpp"
#include "math/matrix.hpp"
#include "ml/svm.hpp"

TEST(SVM, TrainsWithQuadraticSolverAndSeparatesSimpleDataset)
{
    Matrix xTr({
        {2.0, 2.0},
        {2.0, 0.0},
        {0.0, 2.0},
        {0.0, 0.0}
    });

    Matrix yTr({{1.0, 1.0, -1.0, -1.0}});

    ml::SVM model(
        xTr,
        yTr,
        0.0,
        3000,
        1.0,
        DataAugmentationType::NO_OP);

    EXPECT_DOUBLE_EQ(model.score(xTr, yTr), 1.0);

    Matrix positive_query({{2.0, 1.5}});
    Matrix negative_query({{0.1, 0.2}});

    EXPECT_EQ(model.predict(positive_query), 1);
    EXPECT_EQ(model.predict(negative_query), -1);
}
