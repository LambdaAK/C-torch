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

TEST(KNN, ConstructorRejectsInvalidShapes) {
    Matrix xTr({{0.0}, {1.0}});
    Matrix bad_y_rows({{1.0}, {2.0}});
    Matrix bad_y_len({{1.0}});
    Matrix empty_x(0, 1);
    Matrix empty_y(1, 0);

    EXPECT_THROW(ml::KNN(1, xTr, bad_y_rows), std::invalid_argument);
    EXPECT_THROW(ml::KNN(1, xTr, bad_y_len), std::invalid_argument);
    EXPECT_THROW(ml::KNN(1, empty_x, empty_y), std::invalid_argument);
}
