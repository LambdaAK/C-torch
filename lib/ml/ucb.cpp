#include "ucb.hpp"
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>

namespace ml
{

  UCB::UCB(int n_arms)
      : n_arms(n_arms), counts(n_arms, 0), values(n_arms, 0.0), total_count(0) {}

  int UCB::select_arms()
  {
    total_count += 1;

    for (int i = 0; i < n_arms; ++i)
    {
      if (counts[i] == 0)
        return i;
    }

    double log_total = std::log(total_count);
    std::vector<double> ucb_scores(n_arms);

    for (int i = 0; i < n_arms; ++i)
    {
      double bonus = std::sqrt((2 * log_total) / counts[i]);
      ucb_scores[i] = values[i] + bonus;
    }

    auto max_iter = std::max_element(ucb_scores.begin(), ucb_scores.end());
    int argmax = std::distance(ucb_scores.begin(), max_iter);
    return argmax;
  }

  void UCB::update(int arm, double reward)
  {
    counts[arm] += 1;
    int n = counts[arm];
    values[arm] = ((n - 1) * values[arm] + reward) / n;
  }

}