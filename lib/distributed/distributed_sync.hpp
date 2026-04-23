#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include "process_group.hpp"
#include "ml/nn.hpp"

namespace ctorch::distributed
{
/**
 * @brief Broadcasts rank 0's `Sequential` parameters to every worker after topology validation.
 *
 * The helper first synchronizes an architecture signature and then copies each
 * parameter tensor from the root rank to the remaining workers.
 */
inline void broadcast_sequential_parameters(ProcessGroup &group, ::ml::Sequential &model)
{
    const std::size_t parameter_count = model.parameter_count();
    if (parameter_count == 0 || group.world_size() == 1)
    {
        return;
    }

    for (std::size_t index = 0; index < parameter_count; ++index)
    {
        Matrix parameter;
        if (group.rank() == 0)
        {
            if (!model.get_parameter(index, parameter))
            {
                throw std::runtime_error("Unable to read Sequential parameter for broadcast.");
            }
            group.broadcast(parameter, 0);
            continue;
        }

        group.broadcast(parameter, 0);
        if (!model.set_parameter(index, parameter))
        {
            throw std::runtime_error("Unable to write Sequential parameter received from broadcast.");
        }
    }
}

/**
 * @brief Synchronizes a `Sequential` model across workers and rejects topology mismatches.
 *
 * Every worker broadcasts its local architecture signature to rank 0. A collective
 * check then verifies that all workers reported the same signature before the
 * parameter snapshot is exchanged.
 */
inline void synchronize_sequential_model(ProcessGroup &group, ::ml::Sequential &model)
{
    if (group.world_size() == 1)
    {
        return;
    }

    const std::string local_signature = model.architecture_signature();
    std::string canonical_signature = local_signature;
    group.broadcast(canonical_signature, 0);

    Matrix signature_status(1, 1);
    signature_status(0, 0) = canonical_signature == local_signature ? 1.0 : 0.0;
    group.allreduce_sum(signature_status);

    if (signature_status(0, 0) != static_cast<double>(group.world_size()))
    {
        throw std::runtime_error("Sequential architecture mismatch across distributed workers.");
    }

    broadcast_sequential_parameters(group, model);
}

/**
 * @brief All-reduces every gradient buffer owned by a `Sequential` model.
 *
 * The reduced gradients are averaged by world size so the optimizer can keep
 * using local batch-size scaling.
 */
inline void allreduce_sequential_gradients(ProcessGroup &group, ::ml::Sequential &model)
{
    if (group.world_size() == 1)
    {
        return;
    }

    auto parameters = model.parameters();
    if (parameters.empty())
    {
        return;
    }

    const double inv_world_size = 1.0 / static_cast<double>(group.world_size());
    for (auto &parameter : parameters)
    {
        if (!parameter.second)
        {
            throw std::runtime_error("Sequential gradient buffer is null.");
        }

        group.allreduce_sum(*parameter.second);
        *parameter.second = *parameter.second * inv_world_size;
    }
}
} // namespace ctorch::distributed
