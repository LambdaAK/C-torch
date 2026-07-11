#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "math/cuda_backend.hpp"
#include "math/matrix.hpp"

namespace
{
Matrix random_matrix(size_t rows, size_t cols, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-3.0, 3.0);
    Matrix m(rows, cols);
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            m(i, j) = dist(gen);
        }
    }
    return m;
}

void expect_near(const Matrix &gpu, const Matrix &cpu, double tol, const std::string &what)
{
    ASSERT_EQ(gpu.numRows(), cpu.numRows()) << what;
    ASSERT_EQ(gpu.numCols(), cpu.numCols()) << what;
    for (size_t i = 0; i < cpu.numRows(); ++i)
    {
        for (size_t j = 0; j < cpu.numCols(); ++j)
        {
            ASSERT_NEAR(gpu(i, j), cpu(i, j), tol) << what << " at (" << i << ", " << j << ")";
        }
    }
}

/** Runs `op` once on the GPU (thresholds forced to 0) and once on the CPU, then compares. */
void expect_parity(const std::function<Matrix()> &op, double tol, const std::string &what)
{
    ctorch::cuda::set_enabled(true);
    ctorch::cuda::set_thresholds(0, 0);
    const Matrix gpu = op();
    ctorch::cuda::set_enabled(false);
    const Matrix cpu = op();
    ctorch::cuda::set_enabled(true);
    expect_near(gpu, cpu, tol, what);
}

class CudaMatrixParity : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!ctorch::cuda::is_available())
        {
            GTEST_SKIP() << "No CUDA device (or built without -DCTORCH_ENABLE_CUDA=ON)";
        }
    }

    void TearDown() override
    {
        ctorch::cuda::set_enabled(true);
        ctorch::cuda::set_thresholds(size_t{1} << 21, size_t{1} << 20);
    }
};

const std::vector<std::pair<size_t, size_t>> kShapes = {{1, 1}, {1, 17}, {23, 1}, {37, 53}, {256, 256}};
} // namespace

TEST(CudaBackend, ReportsConsistentAvailability)
{
    if (!ctorch::cuda::compiled_with_cuda())
    {
        EXPECT_FALSE(ctorch::cuda::is_available());
        // CPU-only builds must never route work to the GPU, whatever the thresholds.
        ctorch::cuda::set_thresholds(0, 0);
        EXPECT_FALSE(ctorch::cuda::should_use_matmul(64, 64, 64));
        EXPECT_FALSE(ctorch::cuda::should_use_elementwise(1 << 22));
        ctorch::cuda::set_thresholds(size_t{1} << 21, size_t{1} << 20);
    }
}

TEST(CudaBackend, DisabledNeverDispatches)
{
    ctorch::cuda::set_enabled(false);
    ctorch::cuda::set_thresholds(0, 0);
    EXPECT_FALSE(ctorch::cuda::should_use_matmul(512, 512, 512));
    EXPECT_FALSE(ctorch::cuda::should_use_elementwise(1 << 22));
    ctorch::cuda::set_enabled(true);
    ctorch::cuda::set_thresholds(size_t{1} << 21, size_t{1} << 20);
}

TEST_F(CudaMatrixParity, MatmulMatchesCpu)
{
    const std::vector<std::tuple<size_t, size_t, size_t>> dims = {
        {1, 1, 1}, {1, 7, 1}, {5, 1, 9}, {37, 53, 29}, {128, 64, 200}, {256, 256, 256}};
    unsigned seed = 1;
    for (const auto &[m, k, n] : dims)
    {
        const Matrix a = random_matrix(m, k, seed++);
        const Matrix b = random_matrix(k, n, seed++);
        expect_parity([&] { return a * b; }, 1e-9 * static_cast<double>(k),
                      "matmul " + std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n));
    }
}

TEST_F(CudaMatrixParity, ElementwiseAndUnaryMatchCpu)
{
    unsigned seed = 100;
    for (const auto &[rows, cols] : kShapes)
    {
        const Matrix a = random_matrix(rows, cols, seed++);
        const Matrix b = random_matrix(rows, cols, seed++);
        const std::string shape = " " + std::to_string(rows) + "x" + std::to_string(cols);

        expect_parity([&] { return a + b; }, 1e-12, "add" + shape);
        expect_parity([&] { return a - b; }, 1e-12, "sub" + shape);
        expect_parity([&] { return a * 2.5; }, 1e-12, "scale" + shape);
        expect_parity([&] { return a.elm_wise_product(b); }, 1e-12, "mul" + shape);
        expect_parity([&] { return a.transpose(); }, 0.0, "transpose" + shape);
        expect_parity([&] { return a.relu(); }, 0.0, "relu" + shape);
        expect_parity([&] { return a.relu_deriv(); }, 0.0, "relu_deriv" + shape);
        expect_parity([&] { return a.sigmoid(); }, 1e-12, "sigmoid" + shape);
        expect_parity([&] { return a.sigmoid_deriv(); }, 1e-12, "sigmoid_deriv" + shape);
        expect_parity([&] { return a.tanh(); }, 1e-12, "tanh" + shape);
        expect_parity([&] { return a.tanh_deriv(); }, 1e-12, "tanh_deriv" + shape);
    }
}

TEST_F(CudaMatrixParity, ShapeErrorsStillThrow)
{
    ctorch::cuda::set_thresholds(0, 0);
    const Matrix a = random_matrix(3, 4, 7);
    const Matrix b = random_matrix(5, 6, 8);
    EXPECT_THROW(a * b, std::invalid_argument);
    EXPECT_THROW(a + b, std::invalid_argument);
    EXPECT_THROW(a.elm_wise_product(b), std::invalid_argument);
}
