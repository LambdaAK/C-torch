#include <gtest/gtest.h>
#include "math/matrix.hpp"
#include "ml/knn.hpp"

TEST(KNN, PredictsNearestNeighbor) {
    Matrix xTr({{0.0, 0.0}, {10.0, 10.0}, {1.0, 1.0}});
    Matrix yTr({{0.0, 1.0, 2.0}});
    ml::KNN knn(1, xTr, yTr);

    Matrix query({{0.1, 0.1}});
    EXPECT_EQ(knn.predict(query), 0);

    Matrix query2({{9.5, 9.5}});
    EXPECT_EQ(knn.predict(query2), 1);
}

TEST(KNN, TieBreakingUsesSortedNeighborOrder) {
    Matrix xTr({{0.0, 0.0}, {0.0, 0.0}, {5.0, 5.0}});
    Matrix yTr({{1.0, 2.0, 3.0}});
    ml::KNN knn(2, xTr, yTr);
    Matrix q({{0.0, 0.0}});
    int pred = knn.predict(q);
    EXPECT_EQ(pred, 1);
}
