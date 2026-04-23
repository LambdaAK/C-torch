#pragma once

#include <stdexcept>
#include <string>

#include "distributed_sync.hpp"

namespace ctorch::distributed
{
/**
 * @brief Saves a distributed training checkpoint from rank 0.
 *
 * The model parameters and optimizer state are written to `prefix + ".model"`
 * and `prefix + ".optim"` respectively. Non-root ranks no-op.
 */
inline void save_distributed_checkpoint(
    ProcessGroup &group,
    const std::string &prefix,
    ::ml::Sequential &model,
    ::ml::NNOptimizer &optimizer)
{
    if (group.rank() != 0)
    {
        return;
    }

    if (!model.save(prefix + ".model"))
    {
        throw std::runtime_error("Failed to save distributed model checkpoint.");
    }

    if (!optimizer.save_state(prefix + ".optim"))
    {
        throw std::runtime_error("Failed to save distributed optimizer checkpoint.");
    }
}

/**
 * @brief Loads a distributed checkpoint on rank 0 and broadcasts it to the rest of the cluster.
 */
inline void load_distributed_checkpoint(
    ProcessGroup &group,
    const std::string &prefix,
    ::ml::Sequential &model,
    ::ml::NNOptimizer &optimizer)
{
    std::string optimizer_blob;

    if (group.rank() == 0)
    {
        if (!model.load(prefix + ".model"))
        {
            throw std::runtime_error("Failed to load distributed model checkpoint.");
        }

        if (!optimizer.load_state(prefix + ".optim"))
        {
            throw std::runtime_error("Failed to load distributed optimizer checkpoint.");
        }

        optimizer_blob = optimizer.serialize_state();
    }

    synchronize_sequential_model(group, model);
    group.broadcast(optimizer_blob, 0);
    optimizer.deserialize_state(optimizer_blob);
}
} // namespace ctorch::distributed
