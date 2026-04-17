#pragma once

#include "matrix.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace math
{
    enum class LinearProgramSense
    {
        Minimize,
        Maximize
    };

    struct LinearProgramResult
    {
        std::vector<double> solution;
        double objective = 0.0;
        bool optimal = false;
        bool unbounded = false;
        int iterations = 0;
    };

    class LinearProgramSolver
    {
    private:
        int max_iter;
        double tolerance;

        static void pivot(
            std::vector<std::vector<double>> &tableau,
            int pivot_row,
            int pivot_col)
        {
            const int rows = static_cast<int>(tableau.size());
            const int cols = static_cast<int>(tableau.front().size());

            const double pivot_value = tableau[pivot_row][pivot_col];
            if (std::abs(pivot_value) <= 1e-18)
            {
                throw std::runtime_error("Cannot pivot on a near-zero element.");
            }

            for (int j = 0; j < cols; ++j)
            {
                tableau[pivot_row][j] /= pivot_value;
            }

            for (int i = 0; i < rows; ++i)
            {
                if (i == pivot_row)
                {
                    continue;
                }

                const double scale = tableau[i][pivot_col];
                if (std::abs(scale) <= 1e-18)
                {
                    continue;
                }

                for (int j = 0; j < cols; ++j)
                {
                    tableau[i][j] -= scale * tableau[pivot_row][j];
                }
            }
        }

        static double dot(const std::vector<double> &a, const std::vector<double> &b)
        {
            if (a.size() != b.size())
            {
                throw std::invalid_argument("Vector dimensions must match in dot().");
            }

            double result = 0.0;
            for (size_t i = 0; i < a.size(); ++i)
            {
                result += a[i] * b[i];
            }
            return result;
        }

    public:
        LinearProgramSolver(int max_iter = 10000, double tolerance = 1e-9)
            : max_iter(max_iter), tolerance(tolerance)
        {
            if (max_iter <= 0)
            {
                throw std::invalid_argument("max_iter must be positive.");
            }
            if (tolerance <= 0)
            {
                throw std::invalid_argument("tolerance must be positive.");
            }
        }

        // Solves: optimize c^T x, subject to A x <= b, x >= 0.
        LinearProgramResult solve(
            const Matrix &a,
            const std::vector<double> &b,
            const std::vector<double> &c,
            LinearProgramSense sense = LinearProgramSense::Minimize) const
        {
            const int m = static_cast<int>(a.numRows());
            const int n = static_cast<int>(a.numCols());

            if (static_cast<int>(b.size()) != m)
            {
                throw std::invalid_argument("b size must match A row count.");
            }
            if (static_cast<int>(c.size()) != n)
            {
                throw std::invalid_argument("c size must match A column count.");
            }

            for (int i = 0; i < m; ++i)
            {
                if (b[static_cast<size_t>(i)] < -tolerance)
                {
                    throw std::invalid_argument(
                        "Simplex initialization requires b >= 0 for all constraints in this solver variant.");
                }
            }

            const int total_vars = n + m;
            const int rhs_col = total_vars;

            std::vector<std::vector<double>> tableau(
                static_cast<size_t>(m + 1),
                std::vector<double>(static_cast<size_t>(total_vars + 1), 0.0));

            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    tableau[static_cast<size_t>(i)][static_cast<size_t>(j)] = a(static_cast<size_t>(i), static_cast<size_t>(j));
                }

                tableau[static_cast<size_t>(i)][static_cast<size_t>(n + i)] = 1.0;
                tableau[static_cast<size_t>(i)][static_cast<size_t>(rhs_col)] = b[static_cast<size_t>(i)];
            }

            std::vector<double> maximize_c(c.size(), 0.0);
            if (sense == LinearProgramSense::Maximize)
            {
                maximize_c = c;
            }
            else
            {
                for (size_t i = 0; i < c.size(); ++i)
                {
                    maximize_c[i] = -c[i];
                }
            }

            for (int j = 0; j < n; ++j)
            {
                tableau[static_cast<size_t>(m)][static_cast<size_t>(j)] = -maximize_c[static_cast<size_t>(j)];
            }

            std::vector<int> basis(static_cast<size_t>(m), 0);
            for (int i = 0; i < m; ++i)
            {
                basis[static_cast<size_t>(i)] = n + i;
            }

            LinearProgramResult result;
            result.solution.assign(static_cast<size_t>(n), 0.0);

            for (int iter = 0; iter < max_iter; ++iter)
            {
                int entering_col = -1;
                double most_negative = -tolerance;
                for (int j = 0; j < total_vars; ++j)
                {
                    const double reduced_cost = tableau[static_cast<size_t>(m)][static_cast<size_t>(j)];
                    if (reduced_cost < most_negative)
                    {
                        most_negative = reduced_cost;
                        entering_col = j;
                    }
                }

                if (entering_col == -1)
                {
                    result.optimal = true;
                    result.iterations = iter;
                    break;
                }

                int leaving_row = -1;
                double min_ratio = std::numeric_limits<double>::infinity();

                for (int i = 0; i < m; ++i)
                {
                    const double coeff = tableau[static_cast<size_t>(i)][static_cast<size_t>(entering_col)];
                    if (coeff <= tolerance)
                    {
                        continue;
                    }

                    const double rhs = tableau[static_cast<size_t>(i)][static_cast<size_t>(rhs_col)];
                    const double ratio = rhs / coeff;

                    if (ratio < min_ratio - tolerance)
                    {
                        min_ratio = ratio;
                        leaving_row = i;
                    }
                }

                if (leaving_row == -1)
                {
                    result.unbounded = true;
                    result.iterations = iter + 1;
                    return result;
                }

                basis[static_cast<size_t>(leaving_row)] = entering_col;
                pivot(tableau, leaving_row, entering_col);
                result.iterations = iter + 1;
            }

            if (!result.optimal && !result.unbounded)
            {
                throw std::runtime_error("Linear program did not converge within max_iter.");
            }

            for (int i = 0; i < m; ++i)
            {
                const int var = basis[static_cast<size_t>(i)];
                if (var < n)
                {
                    result.solution[static_cast<size_t>(var)] = tableau[static_cast<size_t>(i)][static_cast<size_t>(rhs_col)];
                }
            }

            result.objective = dot(c, result.solution);
            return result;
        }
    };
} // namespace math
