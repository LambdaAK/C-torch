#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <deque>
#include <cmath>

#include "sample.hpp"

class ReplayMemory {
  private:
    std::deque<Sample> memory; // Double-ended queue to store experience samples
    size_t capacity; // Maximum number of samples that can be stored

  public:
    /**
     * Creates a new replay memory buffer with specified capacity
     * @param capacity Maximum number of samples to store
     */
    ReplayMemory(size_t capacity);

    /**
     * Adds a new experience sample to the memory
     * If memory is at capacity, oldest sample is removed
     * @param s Sample to add containing state, action, reward, next state and done flag
     */
    void add(Sample& s);
  
    /**
     * Randomly samples a batch of experiences from memory
     * @param batch_size Number of samples to return
     * @return Vector of randomly selected samples
     */
    std::vector<Sample> sample(size_t batch_size);

    /**
     * Returns current number of samples in memory
     * @return Current size of memory buffer
     */
    size_t size() const;
};