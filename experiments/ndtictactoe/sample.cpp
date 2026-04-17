#include "sample.hpp"

Sample::Sample(std::vector<float>& state, int action, float reward, std::vector<float>& next_state, bool done) : state(state), action(action), reward(reward), next_state(next_state), done(done) {};