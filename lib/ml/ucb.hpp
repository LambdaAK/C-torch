#pragma once

#include <vector>

namespace ml
{

  /**
   * @brief Upper Confidence Bound (UCB1) multi-armed bandit policy.
   *
   * Maintains per-arm empirical means and pull counts to balance
   * exploration and exploitation.
   */
  class UCB
  {
  public:
    /**
     * @brief Creates a UCB policy with a fixed number of arms.
     * @param n_arms Number of available actions (arms).
     */
    UCB(int n_arms);

    /**
     * @brief Selects an arm index according to the UCB criterion.
     * @return Zero-based selected arm index.
     */
    int select_arms();

    /**
     * @brief Updates estimates using observed reward for an arm.
     * @param arm Zero-based arm index.
     * @param reward Observed reward after pulling the arm.
     */
    void update(int arm, double reward);

  private:
    int n_arms;                 ///< Number of available arms.
    int total_count;            ///< Total number of pulls observed so far.
    std::vector<int> counts;    ///< Pull count per arm.
    std::vector<double> values; ///< Estimated mean reward per arm.
  };

}
