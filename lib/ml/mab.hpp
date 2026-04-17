#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

namespace ml
{
    /**
     * @brief Epsilon-greedy multi-armed bandit policy.
     *
     * With probability `eps`, chooses a random arm (exploration);
     * otherwise chooses the arm with the highest estimated value (exploitation).
     */
    class MAB
    {
    private:
        int n_arms;                 ///< Number of available arms.
        float eps;                  ///< Exploration probability.
        std::vector<int> counts;    ///< Number of pulls per arm.
        std::vector<double> values; ///< Running average reward per arm.

    public:
        /**
         * @brief Creates an epsilon-greedy policy.
         * @param n_arms Number of zero-indexed arms.
         * @param eps Probability of selecting a random arm.
         */
        MAB(int n_arms, float eps);

        /**
         * @brief Selects an arm according to epsilon-greedy policy.
         * @return Zero-based arm index.
         */
        int select_arms();

        /**
         * @brief Updates the estimated reward of one arm.
         * @param arm Zero-based arm index.
         * @param reward Observed reward.
         */
        void update(int arm, double reward);

        /**
         * @brief Changes exploration probability at runtime.
         * @param eps New epsilon value.
         */
        void set_epsilon(float eps);
    };
};
