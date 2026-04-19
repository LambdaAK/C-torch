#include <gtest/gtest.h>
#include "reinforce.hpp"

TEST(Reinforce, UpdateLastRewardRequiresStoredTransition) {
    ml::Sequential policy;
    policy.add_layer(std::make_shared<ml::LinearLayer>(1, 1));
    ml::Sequential critic;
    critic.add_layer(std::make_shared<ml::LinearLayer>(1, 1));

    Reinforce reinforce(policy, critic, 0, 0.001f, 0.99f, 1, false, false, true);
    EXPECT_THROW(reinforce.update_last_reward(0, 1.0f), std::logic_error);
    EXPECT_THROW(reinforce.update_last_reward(1, 1.0f), std::out_of_range);
}
