#pragma once

#include "math/matrix.hpp"
#include "ml/nn.hpp"

struct EpisodeStep
{
  std::vector<float> state;
  int action;
  float reward;
  float log_prob;
  Matrix logits;
};

class Reinforce
{
private:
  ml::Sequential agent;
  ml::Sequential critic;
  float baseline;
  float lr;
  float gamma;
  size_t batch_size;
  bool norm_traj;
  bool subtract_baseline;
  bool rew_to_go;

  std::vector<std::vector<EpisodeStep>> episodes;

  ml::NN_SGD optimizer;
  ml::NN_SGD optimizer_critic;

  void store_transition(size_t batch_count, const std::vector<float> &state, int action, float reward, float log_prob, const Matrix &logits);
  void reset_episodes();
  std::vector<float> softmax(Matrix &logits);
  Matrix convert_input(const std::vector<float> &state);

public:
  Reinforce(ml::Sequential policy, ml::Sequential critic, int baseline = 0, float lr = 0.001f, float gamma = 0.99, size_t batch_size = 64, bool norm_traj = false, bool subtract_baseline = false, bool rew_to_go = true);
  int act(size_t batch_count, const std::vector<float> &state, const std::vector<int> &valid_actions, bool inference = false, bool print_logits = false);
  // Called after env.step() to store reward signal for last move
  void update_last_reward(size_t batch_count, float reward);
  // Called after full batch is rolled out to update network
  void update_network();
  void save(const std::string &filename);
  void load(const std::string &filename);
};