#pragma once

#include <algorithm>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

namespace ctorch::parallel
{
inline std::size_t hardware_threads()
{
    const unsigned int count = std::thread::hardware_concurrency();
    return count == 0 ? 2u : static_cast<std::size_t>(count);
}

template <typename Fn>
inline void parallel_for_items(
    std::size_t item_count,
    std::size_t work_units_per_item,
    Fn &&fn,
    std::size_t min_total_work = 16384)
{
    if (item_count == 0)
    {
        return;
    }

    const std::size_t total_work = item_count * work_units_per_item;
    const std::size_t max_threads = hardware_threads();
    if (item_count < 2 || total_work < min_total_work || max_threads < 2)
    {
        fn(0, item_count);
        return;
    }

    const std::size_t workers = std::min(max_threads, item_count);
    std::vector<std::thread> threads;
    threads.reserve(workers - 1);

    const std::size_t base = item_count / workers;
    const std::size_t remainder = item_count % workers;
    std::size_t begin = 0;

    for (std::size_t i = 0; i + 1 < workers; ++i)
    {
        const std::size_t span = base + (i < remainder ? 1 : 0);
        const std::size_t end = begin + span;
        threads.emplace_back([begin, end, &fn]() { fn(begin, end); });
        begin = end;
    }

    fn(begin, item_count);

    for (std::thread &thread : threads)
    {
        thread.join();
    }
}

template <typename T, typename Fn, typename ReduceFn>
inline T parallel_reduce_items(
    std::size_t item_count,
    std::size_t work_units_per_item,
    T identity,
    Fn &&fn,
    ReduceFn &&reduce,
    std::size_t min_total_work = 16384)
{
    if (item_count == 0)
    {
        return identity;
    }

    const std::size_t total_work = item_count * work_units_per_item;
    const std::size_t max_threads = hardware_threads();
    if (item_count < 2 || total_work < min_total_work || max_threads < 2)
    {
        return fn(0, item_count);
    }

    const std::size_t workers = std::min(max_threads, item_count);
    std::vector<T> partials(workers, identity);
    std::vector<std::thread> threads;
    threads.reserve(workers - 1);

    const std::size_t base = item_count / workers;
    const std::size_t remainder = item_count % workers;
    std::size_t begin = 0;

    for (std::size_t i = 0; i + 1 < workers; ++i)
    {
        const std::size_t span = base + (i < remainder ? 1 : 0);
        const std::size_t end = begin + span;
        threads.emplace_back([begin, end, &fn, &partials, i]() { partials[i] = fn(begin, end); });
        begin = end;
    }

    partials[workers - 1] = fn(begin, item_count);

    for (std::thread &thread : threads)
    {
        thread.join();
    }

    T result = identity;
    for (const T &partial : partials)
    {
        result = reduce(std::move(result), partial);
    }

    return result;
}
} // namespace ctorch::parallel
