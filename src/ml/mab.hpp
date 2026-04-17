#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

namespace ml
{
    class MAB
    {
    private:
        int n_arms;
        float eps;
        std::vector<int> counts;
        std::vector<double> values;

    public:
        MAB(int n_arms, float eps);
        // arms are zero-indexed
        int select_arms();
        // arm should be zero-indexed
        void update(int arm, double reward);
        void set_epsilon(float eps);
    };
};
