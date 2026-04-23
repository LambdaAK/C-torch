#pragma once

#include <cstdint>
#include <string>

#include "math/matrix.hpp"

namespace ctorch::distributed
{
/**
 * @brief Abstract collective-communication interface for synchronous training.
 */
class ProcessGroup
{
public:
    virtual ~ProcessGroup() = default;

    /**
     * @brief Returns this worker's rank.
     */
    virtual int rank() const = 0;

    /**
     * @brief Returns the total number of workers.
     */
    virtual int world_size() const = 0;

    /**
     * @brief Synchronizes all workers at a barrier.
     */
    virtual void barrier() = 0;

    /**
     * @brief Broadcasts a string value from the root rank to every worker.
     */
    virtual void broadcast(std::string &value, int root_rank = 0) = 0;

    /**
     * @brief Broadcasts a matrix from the root rank to every worker.
     */
    virtual void broadcast(Matrix &value, int root_rank = 0) = 0;

    /**
     * @brief Sums a matrix across all workers and writes the result back to every rank.
     */
    virtual void allreduce_sum(Matrix &value) = 0;
};
} // namespace ctorch::distributed
