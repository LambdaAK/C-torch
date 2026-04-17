#pragma once

#include <cstddef>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <memory>
#include <deque>
#include <cmath>
#include <string>

#include "replaymemory.hpp"
#include "math/matrix.hpp"
#include "ml/nn.hpp"
#include "ml/nn_optim.hpp"

std::string dqn_optimizer_to_string(ml::NNOptimType optimizer_type);
ml::NNOptimType dqn_optimizer_from_string(const std::string &optimizer_name);

class DQNAgent
{
private:
  ml::Sequential q_network;      // The main network that makes the Q-value predictions
  ml::Sequential target_network; // The target network that is used to compute the target Q-values
  ml::NNOptimizer optimizer;     // Neural network optimizer for training q_network and target_network
  ReplayMemory memory;           // The replay memory that stores the experiences of the agent
  float epsilon;                 // The exploration rate
  float start;                   // The starting exploration rate
  float end;                     // The ending exploration rate
  float decay;                   // The decay rate of the exploration rate
  float gamma;                   // The discount factor
  float lr;                      // The learning rate
  size_t batch_size;             // The batch size for training
  int update_frequency;          // The frequency of updating the target network
  int steps;                     // The number of steps the agent has taken

public:
  /**
   * Creates a new DQN agent
   * @param q_net The Q-network to use for action selection
   * @param target_net The target network to use for Q-value prediction
   * @param start Starting exploration rate
   * @param end Ending exploration rate
   * @param decay Decay rate for exploration
   * @param gamma Discount factor
   * @param lr Learning rate
   * @param batch_size Batch size for training
   * @param memory_capacity Capacity of replay memory
   * @param update_freq Frequency of target network updates
   * @param optimizer_type Optimizer used to train q_network
   */
  DQNAgent(ml::Sequential q_net, ml::Sequential target_net, float start = 0.99f, float end = 0.1f,
           float decay = 0.99995f, float gamma = 0.9f, float lr = 0.001f, size_t batch_size = 64,
           size_t memory_capacity = 10000, int update_freq = 200,
           ml::NNOptimType optimizer_type = ml::NNOptimType::SGD);

  /**
   * Selects an action for the agent to take based on the current state.
   * Uses an epsilon-greedy policy - with probability epsilon, selects a random valid action,
   * otherwise selects the action with the highest predicted Q-value among valid actions.
   *
   * @param state The current state of the environment
   * @param valid_actions Vector of valid action indices that can be taken in the current state
   * @return The selected action index
   */
  int act(const std::vector<float> &state, const std::vector<int> &valid_actions);

  /**
   * Adds a sample to the replay memory.
   *
   * @param s The sample to add to the memory
   */
  void add_to_memory(Sample &s);

  /**
   * Updates the Q-network and target network using a batch of samples from the replay memory.
   *
   * @param batch_size The batch size for training
   */
  void update_networks(size_t batch_size);

  /**
   * Decays the exploration rate (epsilon) towards the minimum value.
   */
  void decay_epsilon();

  /**
   * Gets the current exploration rate.
   */
  float get_epsilon() const;

  /**
   * Sets the exploration rate to a specific value.
   */
  void set_epsilon(float e);

  /**
   * Saves the model weights to a file.
   */
  void save(const std::string &filename);

  /**
   * Loads model weights from a file.
   */
  void load(const std::string &filename);

  /**
   * Converts state vector to Matrix format for NN input.
   */
  Matrix convert_input(const std::vector<float> &state);

  /**
   * Converts output vector to Matrix format for NN output.
   */
  Matrix convert_output(const std::vector<float> &out);
};
