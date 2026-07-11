// CUDA implementation of the Matrix kernels declared in cuda_backend.hpp.
// Every call copies host -> device, runs, and copies device -> host.

#include "cuda_backend.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace ctorch::cuda
{
namespace
{
#define CTORCH_CUDA_CHECK(expr)                                                                \
    do                                                                                         \
    {                                                                                          \
        const cudaError_t ctorch_err_ = (expr);                                                \
        if (ctorch_err_ != cudaSuccess)                                                        \
        {                                                                                      \
            throw std::runtime_error(std::string("CUDA error in ") + #expr + ": " +            \
                                     cudaGetErrorString(ctorch_err_));                         \
        }                                                                                      \
    } while (0)

#define CTORCH_CUBLAS_CHECK(expr)                                                              \
    do                                                                                         \
    {                                                                                          \
        const cublasStatus_t ctorch_status_ = (expr);                                          \
        if (ctorch_status_ != CUBLAS_STATUS_SUCCESS)                                           \
        {                                                                                      \
            throw std::runtime_error(std::string("cuBLAS error in ") + #expr + ": status " +   \
                                     std::to_string(static_cast<int>(ctorch_status_)));        \
        }                                                                                      \
    } while (0)

constexpr int kThreadsPerBlock = 256;

/** RAII device buffer of `n` doubles. */
class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t n) : size_(n)
    {
        CTORCH_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr_), n * sizeof(double)));
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer() { cudaFree(ptr_); }

    double *get() const { return ptr_; }

    void upload(const double *host) const
    {
        CTORCH_CUDA_CHECK(cudaMemcpy(ptr_, host, size_ * sizeof(double), cudaMemcpyHostToDevice));
    }

    void download(double *host) const
    {
        CTORCH_CUDA_CHECK(cudaMemcpy(host, ptr_, size_ * sizeof(double), cudaMemcpyDeviceToHost));
    }

private:
    double *ptr_ = nullptr;
    size_t size_;
};

/** One cuBLAS handle per host thread, since Matrix ops run from worker threads. */
cublasHandle_t cublas_handle()
{
    struct Handle
    {
        cublasHandle_t value = nullptr;
        Handle() { CTORCH_CUBLAS_CHECK(cublasCreate(&value)); }
        ~Handle() { cublasDestroy(value); }
    };
    thread_local Handle handle;
    return handle.value;
}

unsigned int grid_for(size_t n)
{
    const size_t blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
    // Grid-stride loops cover anything beyond this cap.
    return static_cast<unsigned int>(blocks < 65535 ? blocks : 65535);
}

__global__ void unary_kernel(UnaryOp op, const double *in, double *out, size_t n, double scalar)
{
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        const double x = in[i];
        double y = 0.0;
        switch (op)
        {
        case UnaryOp::Scale:
            y = x * scalar;
            break;
        case UnaryOp::Relu:
            y = x > 0.0 ? x : 0.0;
            break;
        case UnaryOp::ReluDeriv:
            y = x <= 0.0 ? 0.0 : 1.0;
            break;
        case UnaryOp::Sigmoid:
            y = 1.0 / (1.0 + exp(-x));
            break;
        case UnaryOp::SigmoidDeriv:
        {
            const double s = 1.0 / (1.0 + exp(-x));
            y = s * (1.0 - s);
            break;
        }
        case UnaryOp::Tanh:
            y = tanh(x);
            break;
        case UnaryOp::TanhDeriv:
        {
            const double t = tanh(x);
            y = 1.0 - t * t;
            break;
        }
        }
        out[i] = y;
    }
}

__global__ void binary_kernel(BinaryOp op, const double *a, const double *b, double *out, size_t n)
{
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        switch (op)
        {
        case BinaryOp::Add:
            out[i] = a[i] + b[i];
            break;
        case BinaryOp::Sub:
            out[i] = a[i] - b[i];
            break;
        case BinaryOp::Mul:
            out[i] = a[i] * b[i];
            break;
        }
    }
}

__global__ void transpose_kernel(const double *in, double *out, size_t rows, size_t cols)
{
    const size_t n = rows * cols;
    for (size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; idx < n;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        const size_t i = idx / cols;
        const size_t j = idx % cols;
        out[j * rows + i] = in[idx];
    }
}

void check_launch()
{
    CTORCH_CUDA_CHECK(cudaGetLastError());
    CTORCH_CUDA_CHECK(cudaDeviceSynchronize());
}
} // namespace

bool compiled_with_cuda()
{
    return true;
}

namespace detail
{
bool device_present()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
} // namespace detail

void matmul(const double *a, const double *b, double *c, size_t m, size_t k, size_t n)
{
    DeviceBuffer d_a(m * k);
    DeviceBuffer d_b(k * n);
    DeviceBuffer d_c(m * n);
    d_a.upload(a);
    d_b.upload(b);

    // cuBLAS is column-major. A row-major buffer is its transpose in column-major,
    // so compute C^T = B^T * A^T, which lands in d_c as row-major C.
    const double alpha = 1.0;
    const double beta = 0.0;
    CTORCH_CUBLAS_CHECK(cublasDgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                                    static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
                                    &alpha, d_b.get(), static_cast<int>(n), d_a.get(),
                                    static_cast<int>(k), &beta, d_c.get(), static_cast<int>(n)));
    d_c.download(c);
}

void unary(UnaryOp op, const double *in, double *out, size_t n, double scalar)
{
    DeviceBuffer d_in(n);
    DeviceBuffer d_out(n);
    d_in.upload(in);
    unary_kernel<<<grid_for(n), kThreadsPerBlock>>>(op, d_in.get(), d_out.get(), n, scalar);
    check_launch();
    d_out.download(out);
}

void binary(BinaryOp op, const double *a, const double *b, double *out, size_t n)
{
    DeviceBuffer d_a(n);
    DeviceBuffer d_b(n);
    DeviceBuffer d_out(n);
    d_a.upload(a);
    d_b.upload(b);
    binary_kernel<<<grid_for(n), kThreadsPerBlock>>>(op, d_a.get(), d_b.get(), d_out.get(), n);
    check_launch();
    d_out.download(out);
}

void transpose(const double *in, double *out, size_t rows, size_t cols)
{
    const size_t n = rows * cols;
    DeviceBuffer d_in(n);
    DeviceBuffer d_out(n);
    d_in.upload(in);
    transpose_kernel<<<grid_for(n), kThreadsPerBlock>>>(d_in.get(), d_out.get(), rows, cols);
    check_launch();
    d_out.download(out);
}
} // namespace ctorch::cuda
