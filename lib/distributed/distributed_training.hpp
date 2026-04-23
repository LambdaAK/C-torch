#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "distributed_sync.hpp"
#include "ml/nn_optim.hpp"
#include "ml/parallel_training.hpp"

namespace ctorch::distributed
{
/**
 * @brief Splits a row-major training matrix evenly across workers.
 *
 * The helper requires the total sample count to be divisible by the world size so
 * every rank receives exactly the same number of rows. That keeps optimizer batch
 * scaling and collective synchronization aligned.
 */
inline std::pair<Matrix, Matrix> equal_row_shard(
    const Matrix &x,
    const Matrix &y,
    int rank,
    int world_size)
{
    if (world_size <= 0)
    {
        throw std::invalid_argument("world_size must be positive.");
    }
    if (rank < 0 || rank >= world_size)
    {
        throw std::invalid_argument("rank must be in [0, world_size).");
    }
    if (y.numRows() != 1)
    {
        throw std::invalid_argument("y must be a row vector.");
    }
    if (x.numRows() != y.numCols())
    {
        throw std::invalid_argument("x and y must describe the same number of samples.");
    }

    const std::size_t total_samples = x.numRows();
    const std::size_t worker_count = static_cast<std::size_t>(world_size);
    if (total_samples % worker_count != 0)
    {
        throw std::invalid_argument("Row count must be divisible by world_size for equal sharding.");
    }

    const std::size_t shard_size = total_samples / worker_count;
    const std::size_t begin = static_cast<std::size_t>(rank) * shard_size;

    Matrix x_shard(shard_size, x.numCols());
    Matrix y_shard(1, shard_size);
    for (std::size_t row = 0; row < shard_size; ++row)
    {
        const std::size_t source_row = begin + row;
        for (std::size_t col = 0; col < x.numCols(); ++col)
        {
            x_shard(row, col) = x(source_row, col);
        }
        y_shard(0, row) = y(0, source_row);
    }

    return {x_shard, y_shard};
}

/**
 * @brief Copies gradient buffers from one `Sequential` model into another.
 */
inline void copy_sequential_gradients(::ml::Sequential &source, ::ml::Sequential &destination)
{
    const auto source_params = source.parameters();
    const auto destination_params = destination.parameters();
    if (source_params.size() != destination_params.size())
    {
        throw std::runtime_error("Sequential gradient layouts do not match.");
    }

    for (std::size_t index = 0; index < source_params.size(); ++index)
    {
        if (!source_params[index].second || !destination_params[index].second)
        {
            throw std::runtime_error("Encountered a null Sequential gradient buffer.");
        }
        if (source_params[index].second->numRows() != destination_params[index].second->numRows() ||
            source_params[index].second->numCols() != destination_params[index].second->numCols())
        {
            throw std::runtime_error("Sequential gradient shapes do not match.");
        }

        *destination_params[index].second = *source_params[index].second;
    }
}

/**
 * @brief Runs a local backward pass, copies gradients into the caller model, and all-reduces them.
 *
 * The caller passes the local shard for the current rank. The helper accumulates
 * gradients on a worker-local copy, copies the resulting gradient buffers back to
 * the caller's model, performs an AllReduce, and then applies the optimizer step.
 */
template <typename Sample, typename ProcessFn>
void distributed_parallel_backpropagate_batch(
    ProcessGroup &group,
    ::ml::Sequential &model,
    ::ml::NNOptimizer &optimizer,
    const std::vector<Sample> &samples,
    ProcessFn &&process_sample)
{
    if (optimizer.get_batch_size() != samples.size())
    {
        throw std::invalid_argument("Optimizer batch size must match the local shard size.");
    }

    optimizer.zero_grad();

    ::ml::Sequential local_model = model.copy_for_parallel_training();
    if (!samples.empty())
    {
        ::ml::parallel_backpropagate_batch(
            local_model,
            samples,
            std::forward<ProcessFn>(process_sample));
    }

    copy_sequential_gradients(local_model, model);
    allreduce_sequential_gradients(group, model);
    optimizer.step();
}
} // namespace ctorch::distributed
