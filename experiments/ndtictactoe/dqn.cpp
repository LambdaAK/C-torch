#include "dqn.hpp"

#include <cctype>
#include <limits>
#include <stdexcept>

namespace
{
std::string to_lower(std::string value)
{
  for (char &ch : value)
  {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}
} // namespace

std::string dqn_optimizer_to_string(ml::NNOptimType optimizer_type)
{
  switch (optimizer_type)
  {
  case ml::NNOptimType::SGD:
    return "sgd";
  case ml::NNOptimType::ADAGRAD:
    return "adagrad";
  case ml::NNOptimType::RMSPROP:
    return "rmsprop";
  case ml::NNOptimType::ADAM:
    return "adam";
  case ml::NNOptimType::ADAMW:
    return "adamw";
  }
  throw std::invalid_argument("Unknown DQN optimizer type.");
}

ml::NNOptimType dqn_optimizer_from_string(const std::string &optimizer_name)
{
  const std::string normalized = to_lower(optimizer_name);
  if (normalized == "sgd")
  {
    return ml::NNOptimType::SGD;
  }
  if (normalized == "adagrad")
  {
    return ml::NNOptimType::ADAGRAD;
  }
  if (normalized == "rmsprop")
  {
    return ml::NNOptimType::RMSPROP;
  }
  if (normalized == "adam")
  {
    return ml::NNOptimType::ADAM;
  }
  if (normalized == "adamw")
  {
    return ml::NNOptimType::ADAMW;
  }

  throw std::invalid_argument("Unsupported optimizer '" + optimizer_name +
                              "'. Expected one of: sgd, adam, adamw, adagrad, rmsprop.");
}
  
DQNAgent::DQNAgent(ml::Sequential q_net, ml::Sequential target_net, float start, float end, float decay, float gamma, 
    float lr, size_t batch_size, size_t memory_capacity, int update_freq, ml::NNOptimType optimizer_type)
    : q_network(q_net), target_network(target_net), memory(memory_capacity), epsilon(start), 
      start(start), end(end), decay(decay), gamma(gamma), lr(lr), batch_size(batch_size), 
      update_frequency(update_freq), steps(0) {
  
  optimizer = ml::NNOptimizer(q_network.parameters(), lr, batch_size, optimizer_type);
  if (!target_network.copy_parameters_from(q_network))
  {
    throw std::runtime_error("Failed to initialize target network from Q-network.");
  }
}

int DQNAgent::act(const std::vector<float> &state, const std::vector<int> &valid_actions)
{
  if (valid_actions.empty())
  {
    throw std::invalid_argument("DQNAgent::act called with no valid actions.");
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  int action;
  if (dis(gen) < epsilon)
  {
    // random exploration: select a random valid action
    std::uniform_int_distribution<> action_dis(0, valid_actions.size() - 1);
    action = valid_actions[action_dis(gen)];
  }
  else
  {
    // exploitation: select action with highest Q-value
    Matrix pred_q = q_network.forward(convert_input(state));
    float max_elt = std::numeric_limits<float>::lowest();
    action = valid_actions[0];
    for (int idx : valid_actions)
    {
      if (pred_q(idx, 0) > max_elt)
      {
        max_elt = pred_q(idx, 0);
        action = idx;
      }
    }
  }

  // make sure the selected action is valid
  if (std::find(valid_actions.begin(), valid_actions.end(), action) == valid_actions.end())
  {
    std::uniform_int_distribution<> action_dis(0, valid_actions.size() - 1);
    action = valid_actions[action_dis(gen)];
  }

  return action;
}

void DQNAgent::add_to_memory(Sample &s)
{
  memory.add(s);
}

void DQNAgent::update_networks(size_t batch_size)
{
  auto batch = memory.sample(batch_size);
  optimizer.zero_grad();

  for (const Sample &elt : batch)
  {
    std::vector<float> state = elt.state;
    int action = elt.action;
    float reward = elt.reward;
    std::vector<float> next_state = elt.next_state;
    bool done = elt.done;

    float target;
    if (done)
    {
      target = reward;
    }
    else
    {
      Matrix next_q_values = target_network.forward(convert_input(next_state));
      float max_elt = next_q_values(0, 0);
      for (size_t i = 0; i < next_q_values.numRows(); ++i)
      {
        if (next_q_values(i, 0) > max_elt)
        {
          max_elt = next_q_values(i, 0);
        }
      }
      target = reward + gamma * max_elt;
    }

    Matrix current_q_values = q_network.forward(convert_input(state));
    Matrix targets = current_q_values;
    targets(action, 0) = target;
    Matrix dL_da = current_q_values - targets;
    q_network.backward(dL_da);
  }

  optimizer.step();
  steps++;
  if (steps % update_frequency == 0)
  {
    if (!target_network.copy_parameters_from(q_network))
    {
      throw std::runtime_error("Failed to copy Q-network parameters to target network.");
    }
  }
}

void DQNAgent::decay_epsilon()
{
  if (epsilon > end)
  {
    epsilon = epsilon * decay;
  }
}

float DQNAgent::get_epsilon() const
{
  return epsilon;
}

void DQNAgent::set_epsilon(float e)
{
  epsilon = e;
}

void DQNAgent::save(const std::string &filename)
{
  q_network.save(filename);
}

void DQNAgent::load(const std::string &filename)
{
  q_network.load(filename);
}

Matrix DQNAgent::convert_input(const std::vector<float> &state)
{
  Matrix input(state.size(), 1);
  for (size_t i = 0; i < state.size(); ++i)
  {
    input(i, 0) = state[i];
  }
  return input;
}

Matrix DQNAgent::convert_output(const std::vector<float> &out)
{
  Matrix output(out.size(), 1);
  for (size_t i = 0; i < out.size(); ++i)
  {
    output(i, 0) = out[i];
  }
  return output;
}
