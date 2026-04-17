#include <gtest/gtest.h>
#include "replaymemory.hpp"

TEST(ReplayMemory, AddAndSample) {
    ReplayMemory m(100);
    std::vector<float> s = {1.f, 2.f, 3.f};
    std::vector<float> n = {1.f, 2.f, 3.f};
    Sample x(s, 0, 1.f, n, false);
    m.add(x);
    EXPECT_EQ(m.size(), 1u);
    std::vector<Sample> batch = m.sample(1);
    ASSERT_EQ(batch.size(), 1u);
    EXPECT_EQ(batch[0].action, 0);
}
