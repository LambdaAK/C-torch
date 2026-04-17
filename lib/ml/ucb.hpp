#pragma once

#include <vector>

namespace ml
{

  class UCB
  {
  public:
    UCB(int n_arms);
    int select_arms();
    void update(int arm, double reward);

  private:
    int n_arms;
    int total_count;
    std::vector<int> counts;
    std::vector<double> values;
  };

}
