#include "replaymemory.hpp"
  
ReplayMemory::ReplayMemory(size_t capacity) : capacity(capacity) {}

// sample: state, action, reward, next_state, done
void ReplayMemory::add(Sample& s) {

  if (memory.size() >= capacity) {
    memory.pop_front();
  }

  memory.push_back(s);
}

std::vector<Sample> ReplayMemory::sample(size_t batch_size) {
  // vector of state, action, reward, next state, done
  std::vector<Sample> batch;
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(memory.begin(), memory.end(), gen);

  for (size_t i = 0; i < std::min(batch_size, memory.size()); ++i) {
    batch.push_back(memory[i]);
  }
  
  return batch;
}

size_t ReplayMemory::size() const {
  return memory.size();
}