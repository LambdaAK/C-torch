#include "mab.hpp"
#include <stdexcept>

namespace ml
{

  MAB::MAB(int n_arms, float eps) : n_arms(n_arms), eps(eps), counts(n_arms, 0), values(n_arms, 0.0)
  {
    if (n_arms <= 0)
    {
      throw std::invalid_argument("MAB: n_arms must be positive.");
    }
    if (eps < 0.0f || eps > 1.0f)
    {
      throw std::invalid_argument("MAB: epsilon must be in [0, 1].");
    }
  }

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

  void MAB::update(int arm, double reward)
  {
    if (arm < 0 || arm >= n_arms)
    {
      throw std::out_of_range("MAB::update arm index out of range.");
    }
    counts[arm] += 1;
    const double n = static_cast<double>(counts[arm]);
    values[arm] += (reward - values[arm]) / n;
  }

  void MAB::set_epsilon(float eps)
  {
    if (eps < 0.0f || eps > 1.0f)
    {
      throw std::invalid_argument("MAB::set_epsilon epsilon must be in [0, 1].");
    }
    this->eps = eps;
  }

}
