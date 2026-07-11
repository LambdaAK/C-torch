// Runtime settings shared by the CUDA and stub backends.

#include "cuda_backend.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace ctorch::cuda
{
namespace
{
bool initial_enabled()
{
    const char *device = std::getenv("CTORCH_DEVICE");
    return !(device != nullptr && std::strcmp(device, "cpu") == 0);
}

std::atomic<bool> g_enabled{initial_enabled()};
// Defaults: ~128^3 multiply-adds for matmul, 1M elements for elementwise ops.
std::atomic<size_t> g_matmul_min_work{size_t{1} << 21};
std::atomic<size_t> g_elementwise_min_elems{size_t{1} << 20};
} // namespace

bool is_available()
{
    static const bool available = compiled_with_cuda() && detail::device_present();
    return available;
}

void set_enabled(bool enabled)
{
    g_enabled.store(enabled);
}

bool is_enabled()
{
    return g_enabled.load();
}

void set_thresholds(size_t matmul_min_work, size_t elementwise_min_elems)
{
    g_matmul_min_work.store(matmul_min_work);
    g_elementwise_min_elems.store(elementwise_min_elems);
}

bool should_use_matmul(size_t m, size_t k, size_t n)
{
    if (!g_enabled.load() || !is_available() || m == 0 || k == 0 || n == 0)
    {
        return false;
    }
    return m * k * n >= g_matmul_min_work.load();
}

bool should_use_elementwise(size_t n)
{
    if (!g_enabled.load() || !is_available() || n == 0)
    {
        return false;
    }
    return n >= g_elementwise_min_elems.load();
}
} // namespace ctorch::cuda
