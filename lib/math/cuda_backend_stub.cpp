// CPU-only build: CUDA is never available, so Matrix never calls the op functions.

#include "cuda_backend.hpp"

#include <stdexcept>

namespace ctorch::cuda
{
namespace
{
[[noreturn]] void not_built()
{
    throw std::runtime_error("ctorch was built without CUDA (configure with -DCTORCH_ENABLE_CUDA=ON)");
}
} // namespace

bool compiled_with_cuda()
{
    return false;
}

namespace detail
{
bool device_present()
{
    return false;
}
} // namespace detail

void matmul(const double *, const double *, double *, size_t, size_t, size_t)
{
    not_built();
}

void unary(UnaryOp, const double *, double *, size_t, double)
{
    not_built();
}

void binary(BinaryOp, const double *, const double *, double *, size_t)
{
    not_built();
}

void transpose(const double *, double *, size_t, size_t)
{
    not_built();
}
} // namespace ctorch::cuda
