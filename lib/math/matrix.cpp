#include "matrix.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <vector>

#ifdef CTORCH_HAS_CUDA
#include "cuda_backend.hpp"
// Route large problems to the GPU; small ones fall through to the CPU path below.
#define CTORCH_CUDA_UNARY(OP, SCALAR)                                                        \
    if (ctorch::cuda::should_use_elementwise(size()))                                       \
    {                                                                                        \
        Matrix gpu_result(rows, cols, Uninitialized{});                                      \
        ctorch::cuda::unary(ctorch::cuda::UnaryOp::OP, data_, gpu_result.data_, size(),      \
                            (SCALAR));                                                       \
        return gpu_result;                                                                   \
    }
#define CTORCH_CUDA_BINARY(OP, OTHER)                                                        \
    if (ctorch::cuda::should_use_elementwise(size()))                                       \
    {                                                                                        \
        Matrix gpu_result(rows, cols, Uninitialized{});                                      \
        ctorch::cuda::binary(ctorch::cuda::BinaryOp::OP, data_, (OTHER).data_,              \
                             gpu_result.data_, size());                                      \
        return gpu_result;                                                                   \
    }
#else
#define CTORCH_CUDA_UNARY(OP, SCALAR)
#define CTORCH_CUDA_BINARY(OP, OTHER)
#endif

namespace
{
// Element counts above which elementwise work is split across the thread pool.
// Cheap ops are memory-bound, so they need far more elements to win than exp/tanh.
constexpr size_t kParallelCheapElems = size_t{1} << 16;
constexpr size_t kParallelTranscendentalElems = size_t{1} << 12;
constexpr size_t kParallelReduceElems = size_t{1} << 17;

/** Applies `op` elementwise: `out[i] = op(in[i])`. Written so the compiler can vectorize it. */
template <typename Op>
void map_unary(const double *in, double *out, size_t n, size_t parallel_threshold, Op op)
{
    ctorch::parallel::parallel_for_items(n, 1, [&](size_t begin, size_t end)
    {
        const double *__restrict src = in;
        double *__restrict dst = out;
        for (size_t i = begin; i < end; ++i)
        {
            dst[i] = op(src[i]);
        }
    }, parallel_threshold);
}

/** Applies `op` elementwise: `out[i] = op(a[i], b[i])`. */
template <typename Op>
void map_binary(const double *a, const double *b, double *out, size_t n, Op op)
{
    ctorch::parallel::parallel_for_items(n, 1, [&](size_t begin, size_t end)
    {
        const double *__restrict lhs = a;
        const double *__restrict rhs = b;
        double *__restrict dst = out;
        for (size_t i = begin; i < end; ++i)
        {
            dst[i] = op(lhs[i], rhs[i]);
        }
    }, kParallelCheapElems);
}

/**
 * Packed, cache-blocked, register-tiled GEMM (the BLIS/GotoBLAS loop structure).
 *
 * For each KC-deep slice of the shared dimension, A is packed into MR-row panels and B
 * into NR-column panels, both zero-padded and contiguous in the order the micro-kernel
 * reads them. The micro-kernel keeps an MR x NR block of C in registers for the whole
 * slice, so each loaded A or B value feeds MR or NR multiply-adds.
 */
namespace gemm
{
constexpr size_t MR = 4;         // micro-tile rows (A panel height)
constexpr size_t NR = 8;         // micro-tile cols (B panel width)
constexpr size_t KC = 256;       // depth of one packed slice
constexpr size_t NC = 4096;      // max columns of B packed at once
constexpr size_t TILE_M = 64;    // max rows of C per parallel task (power of two, >= 16)
constexpr size_t TILE_N = 64;    // max cols of C per parallel task (power of two, >= 16)
constexpr size_t kSmallWork = size_t{1} << 15; // below this m*k*n, packing costs more than it saves

/** Scratch buffer for packed panels; a Matrix gives it aligned, cached storage. */
struct AlignedBuffer
{
    Matrix storage;
    double *ptr;
    explicit AlignedBuffer(size_t count) : storage(1, count), ptr(storage.data()) {}
};

inline size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

/** Packs rows [0, m) x cols [0, kc) of A (row stride lda) into MR-row panels: out[panel][p][i]. */
void pack_a(const double *a, size_t lda, size_t m, size_t kc, double *out)
{
    const size_t panels = ceil_div(m, MR);
    ctorch::parallel::parallel_for_items(panels, MR * kc, [&](size_t begin, size_t end)
    {
        for (size_t panel = begin; panel < end; ++panel)
        {
            double *dst = out + panel * kc * MR;
            const size_t row0 = panel * MR;
            const size_t mr = std::min(MR, m - row0);
            for (size_t i = 0; i < MR; ++i)
            {
                if (i < mr)
                {
                    const double *src = a + (row0 + i) * lda;
                    for (size_t p = 0; p < kc; ++p)
                        dst[p * MR + i] = src[p];
                }
                else
                {
                    for (size_t p = 0; p < kc; ++p)
                        dst[p * MR + i] = 0.0;
                }
            }
        }
    }, size_t{1} << 16);
}

/** Packs rows [0, kc) x cols [0, nc) of B (row stride ldb) into NR-col panels: out[panel][p][j]. */
void pack_b(const double *b, size_t ldb, size_t kc, size_t nc, double *out)
{
    const size_t panels = ceil_div(nc, NR);
    ctorch::parallel::parallel_for_items(panels, NR * kc, [&](size_t begin, size_t end)
    {
        for (size_t panel = begin; panel < end; ++panel)
        {
            double *dst = out + panel * kc * NR;
            const size_t col0 = panel * NR;
            const size_t nr = std::min(NR, nc - col0);
            for (size_t p = 0; p < kc; ++p)
            {
                const double *src = b + p * ldb + col0;
                double *row = dst + p * NR;
                size_t j = 0;
                for (; j < nr; ++j)
                    row[j] = src[j];
                for (; j < NR; ++j)
                    row[j] = 0.0;
            }
        }
    }, size_t{1} << 16);
}

/** C[0:mr, 0:nr] (=|+=) A_panel * B_panel over depth kc. The accumulator stays in registers. */
inline void micro_kernel(size_t kc, const double *__restrict a, const double *__restrict b,
                         double *__restrict c, size_t ldc, size_t mr, size_t nr, bool accumulate)
{
    double acc[MR][NR] = {};
    for (size_t p = 0; p < kc; ++p)
    {
        const double *ap = a + p * MR;
        const double *bp = b + p * NR;
        for (size_t i = 0; i < MR; ++i)
        {
            const double ai = ap[i];
            for (size_t j = 0; j < NR; ++j)
                acc[i][j] += ai * bp[j];
        }
    }

    if (mr == MR && nr == NR)
    {
        for (size_t i = 0; i < MR; ++i)
        {
            double *crow = c + i * ldc;
            if (accumulate)
                for (size_t j = 0; j < NR; ++j)
                    crow[j] += acc[i][j];
            else
                for (size_t j = 0; j < NR; ++j)
                    crow[j] = acc[i][j];
        }
        return;
    }
    for (size_t i = 0; i < mr; ++i)
    {
        double *crow = c + i * ldc;
        for (size_t j = 0; j < nr; ++j)
            crow[j] = accumulate ? crow[j] + acc[i][j] : acc[i][j];
    }
}

/** Direct row-major loop (i-k-j order, contiguous inner loop) for small or skinny products. */
void gemm_small(const double *a, const double *b, double *c, size_t m, size_t k, size_t n)
{
    ctorch::parallel::parallel_for_items(m, k * n, [&](size_t begin, size_t end)
    {
        for (size_t i = begin; i < end; ++i)
        {
            double *__restrict crow = c + i * n;
            const double *arow = a + i * k;
            std::fill(crow, crow + n, 0.0);
            for (size_t p = 0; p < k; ++p)
            {
                const double aip = arow[p];
                const double *__restrict brow = b + p * n;
                for (size_t j = 0; j < n; ++j)
                    crow[j] += aip * brow[j];
            }
        }
    }, size_t{1} << 18);
}

/** Row-major C(m x n) = A(m x k) * B(k x n). Requires k > 0; every element of C is written. */
void gemm(const double *a, const double *b, double *c, size_t m, size_t k, size_t n)
{
    if (m < MR || n < NR || m * k * n < kSmallWork)
    {
        gemm_small(a, b, c, m, k, n);
        return;
    }

    const size_t kc_max = std::min(KC, k);
    const size_t nc_max = std::min(NC, n);
    AlignedBuffer packed_a(ceil_div(m, MR) * MR * kc_max);
    AlignedBuffer packed_b(ceil_div(nc_max, NR) * NR * kc_max);

    // Shrink the per-task tile until there are enough tasks to keep every core busy.
    size_t tile_m = TILE_M;
    size_t tile_n = TILE_N;
    const size_t want_tasks = 2 * ctorch::parallel::hardware_threads();
    while (ceil_div(m, tile_m) * ceil_div(nc_max, tile_n) < want_tasks && (tile_m > 16 || tile_n > 16))
    {
        if (tile_m >= tile_n && tile_m > 16)
            tile_m /= 2;
        else
            tile_n /= 2;
    }

    for (size_t jc = 0; jc < n; jc += NC)
    {
        const size_t nc = std::min(NC, n - jc);
        const size_t b_panels = ceil_div(nc, NR);
        for (size_t pc = 0; pc < k; pc += KC)
        {
            const size_t kc = std::min(KC, k - pc);
            const bool accumulate = pc > 0;
            pack_b(b + pc * n + jc, n, kc, nc, packed_b.ptr);
            pack_a(a + pc, k, m, kc, packed_a.ptr);

            const size_t tiles_m = ceil_div(m, tile_m);
            const size_t tiles_n = ceil_div(nc, tile_n);
            ctorch::parallel::parallel_for_items(tiles_m * tiles_n, tile_m * tile_n * kc,
                [&](size_t begin, size_t end)
            {
                for (size_t tile = begin; tile < end; ++tile)
                {
                    const size_t row_begin = (tile / tiles_n) * tile_m;
                    const size_t row_end = std::min(m, row_begin + tile_m);
                    const size_t panel_begin = (tile % tiles_n) * (tile_n / NR);
                    const size_t panel_end = std::min(b_panels, panel_begin + tile_n / NR);
                    for (size_t row = row_begin; row < row_end; row += MR)
                    {
                        const double *ap = packed_a.ptr + (row / MR) * kc * MR;
                        const size_t mr = std::min(MR, m - row);
                        for (size_t panel = panel_begin; panel < panel_end; ++panel)
                        {
                            const size_t col = panel * NR;
                            micro_kernel(kc, ap, packed_b.ptr + panel * kc * NR,
                                         c + row * n + jc + col, n, mr, std::min(NR, nc - col),
                                         accumulate);
                        }
                    }
                }
            }, size_t{1} << 18);
        }
    }
}
} // namespace gemm
} // namespace

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

namespace
{
/**
 * Caching allocator for large buffers, in the spirit of PyTorch's.
 *
 * Training loops allocate and free same-sized temporaries over and over. For big
 * buffers the system allocator returns memory to the OS on free and page-faults it
 * back in on the next allocation, which costs more than the arithmetic. Freed large
 * buffers are kept here, keyed by exact size, and reused. Small buffers go straight to
 * the system allocator, which is already fast for them.
 */
class BufferCache
{
public:
    static BufferCache &instance()
    {
        static BufferCache *cache = new BufferCache(); // leaked on purpose: outlives static Matrix objects
        return *cache;
    }

    static constexpr size_t kMinCachedBytes = size_t{1} << 16;  // 64 KiB
    static constexpr size_t kMaxCachedBytes = size_t{1} << 28;  // 256 MiB held at most

    void *take(size_t bytes)
    {
        if (bytes >= kMinCachedBytes)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = free_.find(bytes);
            if (it != free_.end() && !it->second.empty())
            {
                void *ptr = it->second.back();
                it->second.pop_back();
                cached_bytes_ -= bytes;
                return ptr;
            }
        }
        return ::operator new(bytes, std::align_val_t{Matrix::kAlignment});
    }

    void give(void *ptr, size_t bytes) noexcept
    {
        if (bytes >= kMinCachedBytes)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (cached_bytes_ + bytes <= kMaxCachedBytes)
            {
                try
                {
                    free_[bytes].push_back(ptr);
                    cached_bytes_ += bytes;
                    return;
                }
                catch (...)
                {
                    // Bookkeeping allocation failed: fall through and release the buffer.
                }
            }
        }
        ::operator delete(ptr, std::align_val_t{Matrix::kAlignment});
    }

private:
    std::mutex mutex_;
    std::unordered_map<size_t, std::vector<void *>> free_;
    size_t cached_bytes_ = 0;
};
} // namespace

double *Matrix::allocate(size_t count)
{
    if (count == 0)
    {
        return nullptr;
    }
    if (count > std::numeric_limits<size_t>::max() / sizeof(double))
    {
        throw std::length_error("Matrix: requested size is too large");
    }
    return static_cast<double *>(BufferCache::instance().take(count * sizeof(double)));
}

void Matrix::deallocate(double *ptr, size_t count) noexcept
{
    if (ptr != nullptr)
    {
        BufferCache::instance().give(ptr, count * sizeof(double));
    }
}

static size_t checked_count(size_t rows, size_t cols)
{
    if (cols != 0 && rows > std::numeric_limits<size_t>::max() / cols)
    {
        throw std::length_error("Matrix: rows * cols overflows");
    }
    return rows * cols;
}

Matrix::Matrix(size_t rows, size_t cols, Uninitialized)
    : rows(rows), cols(cols), data_(allocate(checked_count(rows, cols))) {}

// Constructor: Zero-initialized matrix
Matrix::Matrix(size_t rows, size_t cols) : Matrix(rows, cols, Uninitialized{})
{
    std::fill_n(data_, size(), 0.0);
}

// Constructor: Uses double initializer list to construct the matrix
Matrix::Matrix(std::initializer_list<std::initializer_list<double>> init_list)
    : rows(init_list.size()),
      cols(init_list.size() == 0 ? 0 : init_list.begin()->size())
{
    // Validate before allocating so a ragged list cannot leak.
    for (const auto &row : init_list)
    {
        if (row.size() != cols)
        {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
    }
    data_ = allocate(checked_count(rows, cols));
    size_t i = 0;
    for (const auto &row : init_list)
    {
        std::copy(row.begin(), row.end(), data_ + i * cols);
        ++i;
    }
}

// Default constructor: Creates a 1x1 matrix filled with zero
Matrix::Matrix() : Matrix(1, 1) {}

Matrix::Matrix(const Matrix &other) : Matrix(other.rows, other.cols, Uninitialized{})
{
    std::copy_n(other.data_, size(), data_);
}

Matrix::Matrix(Matrix &&other) noexcept
    : rows(other.rows), cols(other.cols), data_(other.data_)
{
    other.rows = 0;
    other.cols = 0;
    other.data_ = nullptr;
}

Matrix &Matrix::operator=(const Matrix &other)
{
    if (this == &other)
    {
        return *this;
    }
    if (size() != other.size())
    {
        double *fresh = allocate(other.size()); // allocate first: strong exception guarantee
        deallocate(data_, size());
        data_ = fresh;
    }
    rows = other.rows;
    cols = other.cols;
    std::copy_n(other.data_, size(), data_);
    return *this;
}

Matrix &Matrix::operator=(Matrix &&other) noexcept
{
    if (this != &other)
    {
        deallocate(data_, size());
        rows = other.rows;
        cols = other.cols;
        data_ = other.data_;
        other.rows = 0;
        other.cols = 0;
        other.data_ = nullptr;
    }
    return *this;
}

Matrix::~Matrix()
{
    deallocate(data_, size());
}

Matrix Matrix::eye(size_t dim)
{
    Matrix result(dim, dim);
    for (size_t i = 0; i < dim; ++i)
    {
        result.data_[i * dim + i] = 1.0;
    }
    return result;
}

void Matrix::print() const
{
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
            std::cout << data_[i * cols + j] << " ";
        std::cout << "\n";
    }
}

// ---------------------------------------------------------------------------
// Layout ops
// ---------------------------------------------------------------------------

Matrix Matrix::transpose() const
{
#ifdef CTORCH_HAS_CUDA
    if (ctorch::cuda::should_use_elementwise(size()))
    {
        Matrix gpu_result(cols, rows, Uninitialized{});
        ctorch::cuda::transpose(data_, gpu_result.data_, rows, cols);
        return gpu_result;
    }
#endif
    Matrix result(cols, rows, Uninitialized{});
    if (size() == 0)
    {
        return result;
    }

    // Copy in square tiles so both the reads and the strided writes stay in cache.
    constexpr size_t kTile = 32;
    const size_t row_tiles = (rows + kTile - 1) / kTile;
    const double *__restrict src = data_;
    double *__restrict dst = result.data_;
    ctorch::parallel::parallel_for_items(row_tiles, kTile * cols, [&](size_t begin, size_t end)
    {
        for (size_t tile = begin; tile < end; ++tile)
        {
            const size_t i0 = tile * kTile;
            const size_t i1 = std::min(rows, i0 + kTile);
            for (size_t j0 = 0; j0 < cols; j0 += kTile)
            {
                const size_t j1 = std::min(cols, j0 + kTile);
                for (size_t i = i0; i < i1; ++i)
                    for (size_t j = j0; j < j1; ++j)
                        dst[j * rows + i] = src[i * cols + j];
            }
        }
    }, kParallelCheapElems);
    return result;
}

std::vector<Matrix> Matrix::rowsAsMatrices() const
{
    std::vector<Matrix> matrices;
    matrices.reserve(rows);
    for (size_t i = 0; i < rows; ++i)
    {
        Matrix row(1, cols, Uninitialized{});
        std::copy_n(data_ + i * cols, cols, row.data_);
        matrices.push_back(std::move(row));
    }
    return matrices;
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

Matrix Matrix::operator+(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix dimensions must match for addition");
    CTORCH_CUDA_BINARY(Add, other)
    Matrix result(rows, cols, Uninitialized{});
    map_binary(data_, other.data_, result.data_, size(), [](double x, double y) { return x + y; });
    return result;
}

Matrix Matrix::operator-(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    CTORCH_CUDA_BINARY(Sub, other)
    Matrix result(rows, cols, Uninitialized{});
    map_binary(data_, other.data_, result.data_, size(), [](double x, double y) { return x - y; });
    return result;
}

// scalar multiplication
Matrix Matrix::operator*(double scalar) const
{
    CTORCH_CUDA_UNARY(Scale, scalar)
    Matrix result(rows, cols, Uninitialized{});
    map_unary(data_, result.data_, size(), kParallelCheapElems, [scalar](double x) { return x * scalar; });
    return result;
}

// opposite-direction scalar multiplication
Matrix operator*(double scalar, const Matrix &m)
{
    return m * scalar;
}

// matrix multiplication
Matrix Matrix::operator*(const Matrix &m) const
{
    if (cols != m.rows)
        throw std::invalid_argument("Matrix dimensions are incompatible for multiplication: (" +
            std::to_string(rows) + "x" + std::to_string(cols) + ") * (" +
            std::to_string(m.rows) + "x" + std::to_string(m.cols) + ")");

    if (cols == 0 || rows == 0 || m.cols == 0)
    {
        return Matrix(rows, m.cols); // empty inner dimension: all zeros
    }

    Matrix result(rows, m.cols, Uninitialized{});
#ifdef CTORCH_HAS_CUDA
    if (ctorch::cuda::should_use_matmul(rows, cols, m.cols))
    {
        ctorch::cuda::matmul(data_, m.data_, result.data_, rows, cols, m.cols);
        return result;
    }
#endif
    gemm::gemm(data_, m.data_, result.data_, rows, cols, m.cols);
    return result;
}

Matrix Matrix::elm_wise_product(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols)
    {
        throw std::invalid_argument("Matrix dimensions must match for element-wise product");
    }
    CTORCH_CUDA_BINARY(Mul, other)
    Matrix result(rows, cols, Uninitialized{});
    map_binary(data_, other.data_, result.data_, size(), [](double x, double y) { return x * y; });
    return result;
}

// equality check with tolerance
bool Matrix::operator==(const Matrix &other) const
{
    const double EPSILON = 1e-10;
    if (rows != other.rows || cols != other.cols)
        return false;

    for (size_t i = 0; i < size(); ++i)
        if (std::abs(data_[i] - other.data_[i]) > EPSILON)
            return false;
    return true;
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

double Matrix::euclideanDistance(const Matrix &other) const
{
    bool this_is_vector = (rows == 1 || cols == 1);
    bool other_is_vector = (other.rows == 1 || other.cols == 1);

    if (!this_is_vector || !other_is_vector)
        throw std::invalid_argument("Both inputs must be vectors for Euclidean distance");

    const size_t length = size();
    if (length != other.size())
        throw std::invalid_argument("Vectors must have the same length for Euclidean distance");

    if (length == 0)
    {
        return 0.0;
    }

    // Euclidean distance formula: sqrt(sum((a_i - b_i)^2))
    const double *a = data_;
    const double *b = other.data_;
    const double sum_squared_diff = ctorch::parallel::parallel_reduce_items<double>(
        length, 1, 0.0,
        [a, b](size_t begin, size_t end)
        {
            double local_sum = 0.0;
            for (size_t i = begin; i < end; ++i)
            {
                const double diff = a[i] - b[i];
                local_sum += diff * diff;
            }
            return local_sum;
        },
        [](double lhs, double rhs) { return lhs + rhs; },
        kParallelReduceElems);

    return std::sqrt(sum_squared_diff);
}

double Matrix::inner_product(const Matrix &other) const
{
    /*
        One of the following conditions must hold:
        1. Both matrices are row vectors of the same length
        2. Both matrices are column vectors of the same length
    */
    if (rows != other.rows || cols != other.cols || !(rows == 1 || cols == 1))
        throw std::invalid_argument("Dimension mismatch.");

    const size_t length = size();
    if (length == 0)
    {
        return 0.0;
    }

    const double *a = data_;
    const double *b = other.data_;
    return ctorch::parallel::parallel_reduce_items<double>(
        length, 1, 0.0,
        [a, b](size_t begin, size_t end)
        {
            double local_sum = 0.0;
            for (size_t i = begin; i < end; ++i)
            {
                local_sum += a[i] * b[i];
            }
            return local_sum;
        },
        [](double lhs, double rhs) { return lhs + rhs; },
        kParallelReduceElems);
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

Matrix Matrix::relu() const
{
    CTORCH_CUDA_UNARY(Relu, 0.0)
    Matrix result(rows, cols, Uninitialized{});
    map_unary(data_, result.data_, size(), kParallelCheapElems, [](double x) { return std::max(0.0, x); });
    return result;
}

Matrix Matrix::relu_deriv() const
{
    CTORCH_CUDA_UNARY(ReluDeriv, 0.0)
    Matrix result(rows, cols, Uninitialized{});
    map_unary(data_, result.data_, size(), kParallelCheapElems, [](double x) { return x <= 0.0 ? 0.0 : 1.0; });
    return result;
}

Matrix Matrix::sigmoid() const
{
    CTORCH_CUDA_UNARY(Sigmoid, 0.0)
    Matrix result(rows, cols, Uninitialized{});
    map_unary(data_, result.data_, size(), kParallelTranscendentalElems,
              [](double x) { return 1.0 / (1.0 + std::exp(-x)); });
    return result;
}

Matrix Matrix::sigmoid_deriv() const
{
    CTORCH_CUDA_UNARY(SigmoidDeriv, 0.0)
    Matrix result(rows, cols, Uninitialized{});
    map_unary(data_, result.data_, size(), kParallelTranscendentalElems, [](double x)
    {
        const double s = 1.0 / (1.0 + std::exp(-x));
        return s * (1.0 - s);
    });
    return result;
}

Matrix Matrix::tanh() const
{
    CTORCH_CUDA_UNARY(Tanh, 0.0)
    Matrix result(rows, cols, Uninitialized{});
    map_unary(data_, result.data_, size(), kParallelTranscendentalElems, [](double x) { return std::tanh(x); });
    return result;
}

Matrix Matrix::tanh_deriv() const
{
    CTORCH_CUDA_UNARY(TanhDeriv, 0.0)
    Matrix result(rows, cols, Uninitialized{});
    map_unary(data_, result.data_, size(), kParallelTranscendentalElems, [](double x)
    {
        const double t = std::tanh(x);
        return 1.0 - t * t;
    });
    return result;
}

// ---------------------------------------------------------------------------
// Column statistics (kept numerically identical to the original float accumulators)
// ---------------------------------------------------------------------------

Matrix Matrix::l2_norm_cols(const Matrix &X)
{
    std::vector<float> l2_norms(X.numCols(), 0);

    for (size_t i = 0; i < X.numRows(); ++i)
    {
        for (size_t j = 0; j < X.numCols(); ++j)
        {
            l2_norms[j] += X(i, j) * X(i, j);
        }
    }

    constexpr float kEps = 1e-12f;
    Matrix Norm_X = X;
    for (size_t i = 0; i < X.numRows(); ++i)
    {
        for (size_t j = 0; j < X.numCols(); ++j)
        {
            const float denom = std::sqrt(std::max(l2_norms[j], kEps));
            Norm_X(i, j) = Norm_X(i, j) / denom;
        }
    }

    return Norm_X;
}

Matrix Matrix::center_cols(const Matrix &X)
{
    std::vector<float> avgs(X.numCols(), 0);

    for (size_t i = 0; i < X.numRows(); ++i)
    {
        for (size_t j = 0; j < X.numCols(); ++j)
        {
            avgs[j] += X(i, j);
        }
    }

    Matrix Centered_X = X;
    for (size_t i = 0; i < X.numRows(); ++i)
    {
        for (size_t j = 0; j < X.numCols(); ++j)
        {
            Centered_X(i, j) = Centered_X(i, j) - (avgs[j] / X.numRows());
        }
    }

    return Centered_X;
}
