#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "math/parallel.hpp"
#include "nn.hpp"

namespace ml
{
namespace detail
{
inline std::pair<std::size_t, std::size_t> parallel_chunk_bounds(
    std::size_t index,
    std::size_t total_items,
    std::size_t chunk_count)
{
    const std::size_t base = total_items / chunk_count;
    const std::size_t remainder = total_items % chunk_count;
    const std::size_t begin = index * base + std::min(index, remainder);
    const std::size_t span = base + (index < remainder ? 1 : 0);
    return {begin, begin + span};
}
} // namespace detail

/**
 * @brief Runs a batch of training samples in parallel while keeping gradient buffers thread-local.
 *
 * The model parameters remain shared and read-only during the worker phase. Each worker receives
 * its own `Sequential` copy with private gradient buffers, runs a chunk of samples, and then the
 * worker gradients are reduced back into the caller's model.
 *
 * @tparam Sample Sample type stored in the batch.
 * @tparam ProcessFn Callable with signature `void(Sequential&, const Sample&, std::size_t)`.
 * @param model Base model that owns the parameter gradients to be accumulated.
 * @param samples Batch of training samples.
 * @param process_sample Worker callback that performs `forward`/`backward` on a local model copy.
 */
template <typename Sample, typename ProcessFn>
void parallel_backpropagate_batch(
    Sequential &model,
    const std::vector<Sample> &samples,
    ProcessFn &&process_sample)
{
    if (samples.empty())
    {
        return;
    }

    auto base_params = model.parameters();
    if (base_params.empty())
    {
        return;
    }

    const std::size_t worker_count = std::min(ctorch::parallel::hardware_threads(), samples.size());
    if (worker_count < 2)
    {
        for (std::size_t i = 0; i < samples.size(); ++i)
        {
            process_sample(model, samples[i], i);
        }
        return;
    }

    std::vector<Sequential> workers;
    workers.reserve(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i)
    {
        workers.emplace_back(model.copy_for_parallel_training());
    }

    std::exception_ptr error;
    std::mutex error_mutex;

    ctorch::parallel::parallel_for_items(
        worker_count,
        1,
        [&](std::size_t begin, std::size_t end)
        {
            try
            {
                for (std::size_t worker_index = begin; worker_index < end; ++worker_index)
                {
                    const auto [sample_begin, sample_end] =
                        detail::parallel_chunk_bounds(worker_index, samples.size(), worker_count);
                    Sequential &local_model = workers[worker_index];
                    for (std::size_t sample_index = sample_begin; sample_index < sample_end; ++sample_index)
                    {
                        process_sample(local_model, samples[sample_index], sample_index);
                    }
                }
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error)
                {
                    error = std::current_exception();
                }
            }
        },
        2);

    if (error)
    {
        std::rethrow_exception(error);
    }

    std::vector<std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>>> worker_params;
    worker_params.reserve(worker_count);
    for (Sequential &worker : workers)
    {
        worker_params.push_back(worker.parameters());
    }

    ctorch::parallel::parallel_for_items(
        base_params.size(),
        1,
        [&](std::size_t begin, std::size_t end)
        {
            for (std::size_t param_index = begin; param_index < end; ++param_index)
            {
                Matrix &base_grad = *base_params[param_index].second;
                for (const auto &params : worker_params)
                {
                    base_grad = base_grad + *params[param_index].second;
                }
            }
        },
        2);
}
} // namespace ml
