#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <memory>
#include <deque>
#include <cmath>

/**
 * Class representing a single experience/transition sample used for training.
 * These samples are stored in the replay memory buffer and used for experience replay
 * during training of the DQN agent.
 */
class Sample {
  public:
    std::vector<float> state; // The state vector when the action was taken
    int action; // The action that was taken
    float reward; // The reward received for taking the action
    std::vector<float> next_state; // The resulting state after taking the action
    bool done; // Whether this action ended the episode

    Sample(std::vector<float>& state, int action, float reward, std::vector<float>& next_state, bool done);
};