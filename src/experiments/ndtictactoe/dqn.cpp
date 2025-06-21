#include "dqn.hpp"
  
DQNAgent::DQNAgent(ml::Sequential q_net, ml::Sequential target_net, float start, float end, float decay, float gamma, 
    float lr, size_t batch_size, size_t memory_capacity, int update_freq)
    : q_network(q_net), target_network(target_net), memory(memory_capacity), epsilon(start), 
      start(start), end(end), decay(decay), gamma(gamma), lr(lr), batch_size(batch_size), 
      update_frequency(update_freq), steps(0) {
  
  optimizer = ml::NN_SGD(q_network.parameters(), lr, batch_size);
}

int DQNAgent::act(const std::vector<float> &state, const std::vector<int> &valid_actions)
{
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
    float max_elt = pred_q(0, 0);
    size_t max_index = 0;
    for (size_t i = 0; i < pred_q.numRows(); ++i)
    {
      if (pred_q(i, 0) > max_elt)
      {
        max_elt = pred_q(i, 0);
        max_index = i;
      }
    }
    action = max_index;
  }

  // make sure the selected action is valid
  if (epsilon == 0.0f)
  {
    if (std::find(valid_actions.begin(), valid_actions.end(), action) == valid_actions.end())
    {
      std::uniform_int_distribution<> dis(0, valid_actions.size() - 1);
      action = valid_actions[dis(gen)];
    }
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
    target_network = q_network;
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
