#include <gtest/gtest.h>
#include "ml/ucb.hpp"

TEST(UCB, PullsEachArmBeforeUCBFormula) {
    ml::UCB ucb(3);
    int a0 = ucb.select_arms();
    EXPECT_EQ(a0, 0);
    ucb.update(a0, 0.5);
    int a1 = ucb.select_arms();
    EXPECT_EQ(a1, 1);
    ucb.update(a1, 0.5);
    int a2 = ucb.select_arms();
    EXPECT_EQ(a2, 2);
    ucb.update(a2, 0.5);
    for (int i = 0; i < 20; ++i) {
        int a = ucb.select_arms();
        EXPECT_GE(a, 0);
        EXPECT_LT(a, 3);
        ucb.update(a, 0.1 * static_cast<double>(i));
    }
}
