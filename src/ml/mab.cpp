#include "mab.hpp"

namespace ml
{

  MAB::MAB(int n_arms, float eps) : n_arms(n_arms), eps(eps), counts(n_arms, 0), values(n_arms, 0) {}

  int MAB::select_arms()
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<> arm_dis(0, n_arms - 1);

    double random_number = dis(gen);

    if (random_number < eps)
    {
      int random_number = arm_dis(gen);
      return random_number;
    }
    else
    {
      auto max_iter = std::max_element(values.begin(), values.end());
      int argmax = std::distance(values.begin(), max_iter);
      return argmax;
    }
  }

  void MAB::update(int arm, int reward)
  {
    counts[arm] += 1;
    int n = counts[arm];
    values[arm] = ((n_arms - 1) * values[arm] + reward) / n_arms;
  }

  void MAB::set_epsilon(float eps)
  {
    this->eps = eps;
  }

}
