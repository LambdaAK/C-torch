#pragma once

/**
 * @file parallel.hpp
 * @brief Fork-join helpers backed by one persistent, process-wide thread pool.
 *
 * Workers are created once, on first use, and reused for every call, so a
 * parallel region costs a wake-up instead of `std::thread` creation. Nested calls
 * (from inside a parallel region) run inline on the calling thread. An exception
 * thrown by any chunk is rethrown on the caller after all chunks finish.
 */

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace ctorch::parallel
{
namespace detail
{
inline thread_local std::size_t parallel_depth = 0;

class ParallelRegionGuard
{
public:
    ParallelRegionGuard()
    {
        ++parallel_depth;
    }

    ~ParallelRegionGuard()
    {
        --parallel_depth;
    }
};

inline bool in_parallel_region()
{
    return parallel_depth > 0;
}

/**
 * @brief Persistent worker pool. The calling thread also runs chunks, so
 *        `hardware_threads()` threads work in total.
 *
 * The pool is intentionally never destroyed: worker threads stay parked on a
 * condition variable until the process exits, which avoids static-destruction
 * order problems.
 */
class ThreadPool
{
public:
    static ThreadPool &instance()
    {
        static ThreadPool *pool = new ThreadPool(); // leaked on purpose, see class doc
        return *pool;
    }

    /** @brief Runs `task(i)` for every `i` in `[0, chunk_count)` and waits for all of them. */
    template <typename Task>
    void run(std::size_t chunk_count, Task &task)
    {
        Job job;
        job.count = chunk_count;
        job.invoke = [](void *ctx, std::size_t index) { (*static_cast<Task *>(ctx))(index); };
        job.ctx = &task;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            jobs_.push_back(&job);
        }
        work_cv_.notify_all();

        // The caller works on its own job until every chunk has been claimed.
        {
            ParallelRegionGuard guard;
            std::unique_lock<std::mutex> lock(mutex_);
            while (job.next < job.count)
            {
                const std::size_t index = claim(job);
                lock.unlock();
                std::exception_ptr error = execute(job, index);
                lock.lock();
                finish(job, std::move(error));
            }
            done_cv_.wait(lock, [&] { return job.done == job.count; });
        }

        if (job.error)
        {
            std::rethrow_exception(job.error);
        }
    }

    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

private:
    struct Job
    {
        void (*invoke)(void *, std::size_t) = nullptr;
        void *ctx = nullptr;
        std::size_t count = 0;
        std::size_t next = 0; ///< Next unclaimed chunk (guarded by the pool mutex).
        std::size_t done = 0; ///< Finished chunks (guarded by the pool mutex).
        std::exception_ptr error; ///< First exception thrown by a chunk (guarded by the pool mutex).
    };

    ThreadPool()
    {
        const unsigned int hw = std::thread::hardware_concurrency();
        const std::size_t workers = (hw == 0 ? 2u : hw) - 1;
        for (std::size_t i = 0; i < workers; ++i)
        {
            std::thread([this] { worker_loop(); }).detach();
        }
    }

    /** Claims the next chunk of `job`. Caller holds the mutex and `job.next < job.count`. */
    std::size_t claim(Job &job)
    {
        const std::size_t index = job.next++;
        if (job.next == job.count)
        {
            // Fully claimed: no other thread needs to find it any more.
            jobs_.erase(std::find(jobs_.begin(), jobs_.end(), &job));
        }
        return index;
    }

    /** Runs one chunk without holding the mutex; returns its exception, if any. */
    static std::exception_ptr execute(Job &job, std::size_t index)
    {
        try
        {
            job.invoke(job.ctx, index);
        }
        catch (...)
        {
            return std::current_exception();
        }
        return nullptr;
    }

    /** Marks one chunk finished. Caller holds the mutex. After this a worker must not touch `job`. */
    void finish(Job &job, std::exception_ptr error)
    {
        if (error && !job.error)
        {
            job.error = std::move(error);
        }
        if (++job.done == job.count)
        {
            done_cv_.notify_all();
        }
    }

    void worker_loop()
    {
        ++parallel_depth; // work run by the pool never spawns nested parallel regions
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;)
        {
            work_cv_.wait(lock, [&] { return !jobs_.empty(); });
            Job &job = *jobs_.front();
            const std::size_t index = claim(job);
            lock.unlock();
            std::exception_ptr error = execute(job, index);
            lock.lock();
            finish(job, std::move(error));
        }
    }

    std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    std::deque<Job *> jobs_;
};
} // namespace detail

inline std::size_t hardware_threads()
{
    static const std::size_t count = []
    {
        const unsigned int hw = std::thread::hardware_concurrency();
        return hw == 0 ? std::size_t{2} : static_cast<std::size_t>(hw);
    }();
    return count;
}

/**
 * @brief Splits `[0, item_count)` into contiguous ranges and calls `fn(begin, end)` on each.
 *
 * Runs inline when the total work (`item_count * work_units_per_item`) is below
 * `min_total_work`, when there is one item, or when already inside a parallel region.
 */
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
    if (item_count < 2 || total_work < min_total_work || max_threads < 2 || detail::in_parallel_region())
    {
        fn(0, item_count);
        return;
    }

    const std::size_t workers = std::min(max_threads, item_count);
    const std::size_t base = item_count / workers;
    const std::size_t remainder = item_count % workers;
    auto task = [&](std::size_t chunk)
    {
        const std::size_t begin = chunk * base + std::min(chunk, remainder);
        const std::size_t end = begin + base + (chunk < remainder ? 1 : 0);
        fn(begin, end);
    };
    detail::ThreadPool::instance().run(workers, task);
}

/**
 * @brief Parallel map-reduce over `[0, item_count)`. Partials are combined in chunk order,
 *        so the result is deterministic for a given thread count.
 */
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
    if (item_count < 2 || total_work < min_total_work || max_threads < 2 || detail::in_parallel_region())
    {
        return fn(0, item_count);
    }

    const std::size_t workers = std::min(max_threads, item_count);
    const std::size_t base = item_count / workers;
    const std::size_t remainder = item_count % workers;
    std::vector<T> partials(workers, identity);
    auto task = [&](std::size_t chunk)
    {
        const std::size_t begin = chunk * base + std::min(chunk, remainder);
        const std::size_t end = begin + base + (chunk < remainder ? 1 : 0);
        partials[chunk] = fn(begin, end);
    };
    detail::ThreadPool::instance().run(workers, task);

    T result = identity;
    for (const T &partial : partials)
    {
        result = reduce(std::move(result), partial);
    }
    return result;
}
} // namespace ctorch::parallel
