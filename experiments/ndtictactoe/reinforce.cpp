#include "reinforce.hpp"
#include <stdexcept>

Reinforce::Reinforce(ml::Sequential policy, ml::Sequential critic, int baseline, float lr, float gamma, size_t batch_size, bool norm_traj, bool subtract_baseline, bool rew_to_go)
    : agent(policy), critic(critic), baseline(baseline), lr(lr), gamma(gamma), batch_size(batch_size), episodes(batch_size),
      norm_traj(norm_traj), subtract_baseline(subtract_baseline), rew_to_go(rew_to_go)
{
  optimizer = ml::NN_SGD(policy.parameters(), 0.0001, batch_size);
  optimizer_critic = ml::NN_SGD(critic.parameters(), 0.0005, batch_size);
}

int Reinforce::act(size_t batch_count, const std::vector<float> &state, const std::vector<int> &valid_actions, bool inference, bool print_logits)
{
  if (batch_count >= episodes.size())
  {
    throw std::out_of_range("Reinforce::act batch_count out of range.");
  }
  if (valid_actions.empty())
  {
    throw std::invalid_argument("Reinforce::act requires at least one valid action.");
  }

  Matrix logits = agent.forward(convert_input(state));

  // mask for invalid actions (set to a large negative value)
  for (int i = 0; i < logits.numRows(); ++i)
  {
    if (std::find(valid_actions.begin(), valid_actions.end(), i) == valid_actions.end())
    {
      logits(i, 0) = -1e9;
    }
  }

  std::vector<float> probs = softmax(logits);

  if (print_logits)
  {
    std::cout << "Raw logits: ";
    for (int i = 0; i < logits.numRows(); ++i)
    {
      if (std::find(valid_actions.begin(), valid_actions.end(), i) != valid_actions.end())
      {
        std::cout << logits(i, 0) << " ";
      }
      else
      {
        std::cout << "X ";
      }
    }
    std::cout << std::endl;

    std::cout << "Probabilities: ";
    for (int i = 0; i < probs.size(); ++i)
    {
      if (std::find(valid_actions.begin(), valid_actions.end(), i) != valid_actions.end())
      {
        std::cout << probs[i] << " ";
      }
      else
      {
        std::cout << "X ";
      }
    }
    std::cout << std::endl;
  }

  if (inference)
  {
    auto max_it = std::max_element(probs.begin(), probs.end());
    int action = std::distance(probs.begin(), max_it);

    if (std::find(valid_actions.begin(), valid_actions.end(), action) == valid_actions.end())
    {
      action = valid_actions[0];
    }
    return action;
  }

  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  std::mt19937 gen(std::random_device{}());
  int action = dist(gen);

  float log_prob = std::log(probs[action] + 1e-10); 

  store_transition(batch_count, state, action, 0.0f, log_prob, logits);

  if (std::find(valid_actions.begin(), valid_actions.end(), action) == valid_actions.end())
  {
    std::uniform_int_distribution<> dis(0, valid_actions.size() - 1);
    action = valid_actions[dis(gen)];
  }

  return action;
}

void Reinforce::update_last_reward(size_t batch_count, float reward)
{
  if (batch_count >= episodes.size())
  {
    throw std::out_of_range("Reinforce::update_last_reward batch_count out of range.");
  }
  if (episodes[batch_count].empty())
  {
    throw std::logic_error("Reinforce::update_last_reward called before any transition was stored.");
  }
  episodes[batch_count].back().reward = reward;
}

void Reinforce::store_transition(size_t batch_count, const std::vector<float> &state, int action, float reward, float log_prob, const Matrix &logits)
{
  if (batch_count >= episodes.size())
  {
    throw std::out_of_range("Reinforce::store_transition batch_count out of range.");
  }
  episodes[batch_count].push_back({state, action, reward, log_prob, logits});
}

std::vector<float> Reinforce::softmax(Matrix &logits)
{
  std::vector<float> probs(logits.numRows());

  float max_val = -std::numeric_limits<float>::max();
  for (size_t i = 0; i < probs.size(); ++i)
  {
    if (logits(i, 0) > max_val && logits(i, 0) != -1e9)
    {
      max_val = logits(i, 0);
    }
  }

  float sum_exp = 0.0f;
  for (size_t i = 0; i < probs.size(); ++i)
  {
    if (logits(i, 0) == -1e9)
    {
      probs[i] = 0.0f;
    }
    else
    {
      probs[i] = std::exp(logits(i, 0) - max_val);
      sum_exp += probs[i];
    }
  }

  for (float &p : probs)
  {
    p /= (sum_exp + 1e-10);
  }

  return probs;
}

void Reinforce::update_network()
{
  optimizer.zero_grad();
  optimizer_critic.zero_grad();

  float total_policy_loss = 0.0f;
  float total_critic_loss = 0.0f;
  int total_steps = 0;

  float total_entropy = 0.0f;

  for (size_t batch_count = 0; batch_count < batch_size; ++batch_count)
  {
    if (episodes[batch_count].empty())
    {
      continue; // skip empty episodes
    }

    std::vector<float> returns(episodes[batch_count].size());
    float G = 0.0f;

    for (int i = episodes[batch_count].size() - 1; i >= 0; --i)
    {
      G = episodes[batch_count][i].reward + gamma * G;
      returns[i] = G;
    }

    if (norm_traj && returns.size() > 1)
    {
      float mean = std::accumulate(returns.begin(), returns.end(), 0.0f) / returns.size();
      float sq_sum = 0.0f;
      for (float r : returns)
      {
        sq_sum += r * r;
      }
      float std_dev = std::sqrt(sq_sum / returns.size() - mean * mean);

      if (std_dev > 1e-8f)
      {
        for (float &r : returns)
        {
          r = (r - mean) / (std_dev + 1e-8f);
        }
      }
    }

    for (size_t i = 0; i < episodes[batch_count].size(); ++i)
    {
      auto &step = episodes[batch_count][i];

      Matrix state_input = convert_input(step.state);
      Matrix value_pred = critic.forward(state_input);
      float value = value_pred(0, 0);

      float target_return = rew_to_go ? returns[i] : returns[0];

      float advantage = target_return;
      if (subtract_baseline)
      {
        advantage -= value;
      }

      // Policy Gradient Update
      Matrix policy_grad = Matrix(step.logits.numRows(), 1);
      std::vector<float> probs = softmax(step.logits);

      float step_entropy = 0.0f;
      for (float p : probs)
      {
        if (p > 1e-8f)
        {
          step_entropy -= p * std::log(p);
        }
      }

      total_entropy += step_entropy;

      for (int a = 0; a < policy_grad.numRows(); ++a)
      {
        if (a == step.action)
        {
          policy_grad(a, 0) = advantage * (1.0f - probs[a]);
        }
        else
        {
          policy_grad(a, 0) = advantage * (-probs[a]);
        }
      }

      policy_grad = -1.0f * policy_grad;
      agent.backward(policy_grad);

      total_policy_loss += -advantage * std::log(std::max(probs[step.action], 1e-8f));

      // Critic Loss
      float critic_loss = (value - target_return) * (value - target_return);
      total_critic_loss += critic_loss;

      Matrix critic_grad(1, 1);
      critic_grad(0, 0) = 2.0f * (value - target_return);
      critic.backward(critic_grad);

      total_steps++;
    }
  }

  optimizer.step();
  optimizer_critic.step();
  reset_episodes();

  // Logging
  static int update_count = 0;
  update_count++;

  const int log_interval = 100;

  if (update_count % log_interval == 0 && total_steps > 0)
  {
    std::cout << "[Update " << update_count << "] "
              << "Avg Policy Loss: " << (total_policy_loss / total_steps)
              << " | Avg Critic Loss: " << (total_critic_loss / total_steps)
              << " | Avg Entropy: " << (total_entropy / total_steps)
              << " | Steps: " << total_steps << std::endl;
  }
}

void Reinforce::reset_episodes()
{
  episodes = std::vector<std::vector<EpisodeStep>>(batch_size);
}

Matrix Reinforce::convert_input(const std::vector<float> &state)
{
  Matrix input(state.size(), 1);
  for (size_t i = 0; i < state.size(); ++i)
    input(i, 0) = state[i];
  return input;
}

void Reinforce::save(const std::string &filename)
{
  agent.save(filename);
}

void Reinforce::load(const std::string &filename)
{
  agent.load(filename);
}
