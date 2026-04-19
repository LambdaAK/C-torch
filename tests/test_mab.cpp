#include <gtest/gtest.h>
#include "ml/mab.hpp"

TEST(MAB, IncrementalMeanFavorsHigherRewardArm) {
    ml::MAB mab(2, 0.0f);
    mab.update(0, 10.0);
    mab.update(1, 20.0);
    mab.set_epsilon(0.0f);
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(mab.select_arms(), 1);
    }
}

TEST(MAB, EpsilonGreedyExplores) {
    ml::MAB mab(3, 1.0f);
    for (int i = 0; i < 50; ++i) {
        int a = mab.select_arms();
        EXPECT_GE(a, 0);
        EXPECT_LT(a, 3);
    }
}

TEST(MAB, RejectsInvalidArmCount) {
    EXPECT_THROW(ml::MAB(0, 0.1f), std::invalid_argument);
}

TEST(MAB, UpdateRejectsOutOfRangeArm) {
    ml::MAB mab(2, 0.0f);
    EXPECT_THROW(mab.update(-1, 1.0), std::out_of_range);
    EXPECT_THROW(mab.update(2, 1.0), std::out_of_range);
}
