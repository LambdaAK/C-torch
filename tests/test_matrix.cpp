#include <gtest/gtest.h>
#include <cmath>
#include "math/matrix.hpp"

TEST(Matrix, DefaultConstructorIsOneByOneZero) {
    Matrix m;
    EXPECT_EQ(m.numRows(), 1u);
    EXPECT_EQ(m.numCols(), 1u);
    EXPECT_DOUBLE_EQ(m(0, 0), 0.0);
}

TEST(Matrix, InnerProductRowVectors) {
    Matrix a({{1.0, 2.0, 3.0}});
    Matrix b({{4.0, 5.0, 6.0}});
    EXPECT_DOUBLE_EQ(a.inner_product(b), 32.0);
}

TEST(Matrix, Transpose) {
    Matrix m({{1.0, 2.0}, {3.0, 4.0}});
    Matrix t = m.transpose();
    EXPECT_DOUBLE_EQ(t(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(t(0, 1), 3.0);
    EXPECT_DOUBLE_EQ(t(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(t(1, 1), 4.0);
}

TEST(Matrix, CopyAssignment) {
    Matrix a({{1.0, 2.0}});
    Matrix b;
    b = a;
    EXPECT_DOUBLE_EQ(b(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(b(0, 1), 2.0);
    b(0, 0) = 9.0;
    EXPECT_DOUBLE_EQ(a(0, 0), 1.0);
}

TEST(Matrix, EmptyInitializerListCreatesZeroByZero) {
    Matrix m({});
    EXPECT_EQ(m.numRows(), 0u);
    EXPECT_EQ(m.numCols(), 0u);
}

TEST(Matrix, LargeMatrixMultiplicationProducesExpectedValues) {
    Matrix a(64, 64);
    Matrix b(64, 64);

    for (size_t i = 0; i < 64; ++i) {
        for (size_t j = 0; j < 64; ++j) {
            a(i, j) = 1.0;
            b(i, j) = 2.0;
        }
    }

    Matrix result = a * b;

    EXPECT_EQ(result.numRows(), 64u);
    EXPECT_EQ(result.numCols(), 64u);
    for (size_t i = 0; i < 64; ++i) {
        for (size_t j = 0; j < 64; ++j) {
            EXPECT_DOUBLE_EQ(result(i, j), 128.0);
        }
    }
}

TEST(Matrix, LargeVectorInnerProductAndDistance) {
    Matrix a(1, 8192);
    Matrix b(1, 8192);

    for (size_t i = 0; i < 8192; ++i) {
        a(0, i) = 1.0;
        b(0, i) = 2.0;
    }

    EXPECT_DOUBLE_EQ(a.inner_product(b), 16384.0);
    EXPECT_NEAR(a.euclideanDistance(b), std::sqrt(8192.0), 1e-9);
}

// ---------------------------------------------------------------------------
// Storage semantics and kernel correctness for the raw-buffer Matrix
// ---------------------------------------------------------------------------

#include <atomic>
#include <cstdint>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "math/parallel.hpp"

namespace
{
Matrix random_matrix(size_t rows, size_t cols, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Matrix m(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            m(i, j) = dist(gen);
    return m;
}

Matrix naive_matmul(const Matrix &a, const Matrix &b)
{
    Matrix c(a.numRows(), b.numCols());
    for (size_t i = 0; i < a.numRows(); ++i)
        for (size_t j = 0; j < b.numCols(); ++j)
        {
            long double sum = 0.0L;
            for (size_t k = 0; k < a.numCols(); ++k)
                sum += static_cast<long double>(a(i, k)) * b(k, j);
            c(i, j) = static_cast<double>(sum);
        }
    return c;
}
} // namespace

TEST(MatrixStorage, BufferIsAlignedAndZeroed) {
    Matrix m(13, 7);
    ASSERT_NE(m.data(), nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(m.data()) % Matrix::kAlignment, 0u);
    EXPECT_EQ(m.size(), 91u);
    for (size_t i = 0; i < m.size(); ++i)
        EXPECT_EQ(m.data()[i], 0.0);
}

TEST(MatrixStorage, MoveStealsBufferAndEmptiesSource) {
    Matrix a = random_matrix(5, 6, 1);
    const double *buffer = a.data();
    Matrix b(std::move(a));
    EXPECT_EQ(b.data(), buffer);
    EXPECT_EQ(a.numRows(), 0u);
    EXPECT_EQ(a.numCols(), 0u);
    EXPECT_EQ(a.data(), nullptr);

    Matrix c(2, 2);
    c = std::move(b);
    EXPECT_EQ(c.data(), buffer);
    EXPECT_EQ(c.numRows(), 5u);
    EXPECT_EQ(c.numCols(), 6u);
}

TEST(MatrixStorage, CopyIsDeepAndHandlesShapeChanges) {
    Matrix a = random_matrix(4, 3, 2);
    Matrix b(a);
    EXPECT_NE(b.data(), a.data());
    EXPECT_TRUE(a == b);
    b(0, 0) += 1.0;
    EXPECT_FALSE(a == b);

    Matrix c(1, 1);
    c = a; // grow
    EXPECT_TRUE(c == a);
    Matrix d = random_matrix(3, 4, 3);
    d = a; // same element count, different shape: reuses the buffer
    EXPECT_EQ(d.numRows(), 4u);
    EXPECT_EQ(d.numCols(), 3u);
    EXPECT_TRUE(d == a);
    Matrix &alias = d;
    d = alias; // self-assignment
    EXPECT_TRUE(d == a);
}

TEST(MatrixStorage, EmptyShapesWork) {
    Matrix a(0, 5);
    EXPECT_EQ(a.size(), 0u);
    EXPECT_EQ(a.data(), nullptr);
    Matrix b(5, 0);
    Matrix prod = a * Matrix(5, 3);
    EXPECT_EQ(prod.numRows(), 0u);
    EXPECT_EQ(prod.numCols(), 3u);
    Matrix zeros = b * Matrix(0, 4); // empty inner dimension -> 5x4 zeros
    EXPECT_TRUE(zeros == Matrix(5, 4));
    EXPECT_EQ(a.transpose().numRows(), 5u);
}

TEST(MatrixKernels, MatmulMatchesNaiveAcrossShapes) {
    // Covers the small path, micro-tile edges (not multiples of 4x8), multiple
    // 256-deep slices, and more than one 4096-column block of B.
    const std::vector<std::tuple<size_t, size_t, size_t>> shapes = {
        {1, 1, 1}, {3, 5, 7}, {4, 8, 8}, {17, 33, 9}, {64, 64, 64}, {65, 257, 63},
        {130, 300, 70}, {200, 513, 31}, {9, 40, 4100}, {256, 256, 256}, {1, 600, 600}, {600, 600, 1}};
    unsigned seed = 10;
    for (const auto &[m, k, n] : shapes) {
        const Matrix a = random_matrix(m, k, seed++);
        const Matrix b = random_matrix(k, n, seed++);
        const Matrix got = a * b;
        const Matrix want = naive_matmul(a, b);
        ASSERT_EQ(got.numRows(), m);
        ASSERT_EQ(got.numCols(), n);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                ASSERT_NEAR(got(i, j), want(i, j), 1e-11 * static_cast<double>(k))
                    << m << "x" << k << "x" << n << " at (" << i << "," << j << ")";
    }
}

TEST(MatrixKernels, TransposeLargeAndRagged) {
    for (const auto &[r, c] : std::vector<std::pair<size_t, size_t>>{{1, 1}, {1, 100}, {33, 65}, {700, 300}}) {
        const Matrix a = random_matrix(r, c, static_cast<unsigned>(r * 1000 + c));
        const Matrix t = a.transpose();
        ASSERT_EQ(t.numRows(), c);
        ASSERT_EQ(t.numCols(), r);
        for (size_t i = 0; i < r; ++i)
            for (size_t j = 0; j < c; ++j)
                ASSERT_EQ(t(j, i), a(i, j));
    }
}

TEST(MatrixKernels, LargeElementwiseOpsMatchScalarMath) {
    const Matrix a = random_matrix(400, 300, 77); // big enough to run on the thread pool
    const Matrix b = random_matrix(400, 300, 78);
    const Matrix sum = a + b, diff = a - b, prod = a.elm_wise_product(b), scaled = a * -1.5;
    const Matrix sig = a.sigmoid(), th = a.tanh(), rl = a.relu();
    for (size_t i = 0; i < 400; ++i)
        for (size_t j = 0; j < 300; ++j) {
            const double x = a(i, j), y = b(i, j);
            ASSERT_DOUBLE_EQ(sum(i, j), x + y);
            ASSERT_DOUBLE_EQ(diff(i, j), x - y);
            ASSERT_DOUBLE_EQ(prod(i, j), x * y);
            ASSERT_DOUBLE_EQ(scaled(i, j), x * -1.5);
            ASSERT_DOUBLE_EQ(sig(i, j), 1.0 / (1.0 + std::exp(-x)));
            ASSERT_DOUBLE_EQ(th(i, j), std::tanh(x));
            ASSERT_DOUBLE_EQ(rl(i, j), std::max(0.0, x));
        }
}

TEST(ParallelPool, CoversEveryItemExactlyOnce) {
    std::vector<std::atomic<int>> hits(100000);
    ctorch::parallel::parallel_for_items(hits.size(), 1, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i)
            hits[i].fetch_add(1);
    }, 1);
    for (const auto &h : hits)
        ASSERT_EQ(h.load(), 1);
}

TEST(ParallelPool, PropagatesExceptionsToCaller) {
    EXPECT_THROW(ctorch::parallel::parallel_for_items(1000, 1000, [](size_t begin, size_t) {
        if (begin > 0)
            throw std::runtime_error("boom");
    }, 1), std::runtime_error);
    // The pool must still work afterwards.
    const double total = ctorch::parallel::parallel_reduce_items<double>(
        1000, 1000, 0.0,
        [](size_t begin, size_t end) { return static_cast<double>(end - begin); },
        [](double x, double y) { return x + y; }, 1);
    EXPECT_EQ(total, 1000.0);
}
