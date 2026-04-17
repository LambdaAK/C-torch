#pragma once

#include "matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace math
{
    struct QuadraticProgramResult
    {
        std::vector<double> solution;
        double objective = 0.0;
        bool converged = false;
        int iterations = 0;
    };

    class QuadraticProgramSolver
    {
    private:
        int max_iter;
        double tolerance;
        int projection_iters;
        double step_size;

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

        static std::vector<double> mat_vec(const Matrix &m, const std::vector<double> &x)
        {
            if (m.numCols() != x.size())
            {
                throw std::invalid_argument("Matrix/vector dimensions do not align in mat_vec().");
            }

            std::vector<double> result(m.numRows(), 0.0);
            for (size_t i = 0; i < m.numRows(); ++i)
            {
                for (size_t j = 0; j < m.numCols(); ++j)
                {
                    result[i] += m(i, j) * x[j];
                }
            }
            return result;
        }

        static void clip_to_bounds(
            std::vector<double> &x,
            const std::vector<double> &lower_bounds,
            const std::vector<double> &upper_bounds)
        {
            for (size_t i = 0; i < x.size(); ++i)
            {
                x[i] = std::clamp(x[i], lower_bounds[i], upper_bounds[i]);
            }
        }

        std::vector<double> project_to_feasible_set(
            const std::vector<double> &x,
            const std::vector<double> &lower_bounds,
            const std::vector<double> &upper_bounds,
            const std::vector<double> &equality_coeffs,
            double equality_value) const
        {
            std::vector<double> projected = x;

            const bool has_equality = !equality_coeffs.empty();
            double equality_norm_sq = 0.0;
            if (has_equality)
            {
                equality_norm_sq = dot(equality_coeffs, equality_coeffs);
                if (equality_norm_sq <= 1e-18)
                {
                    if (std::abs(equality_value) > tolerance)
                    {
                        throw std::invalid_argument("Equality constraint is infeasible because coefficients are all zero while target is non-zero.");
                    }
                }
            }

            for (int iter = 0; iter < projection_iters; ++iter)
            {
                if (has_equality && equality_norm_sq > 1e-18)
                {
                    const double residual = dot(equality_coeffs, projected) - equality_value;
                    const double scale = residual / equality_norm_sq;
                    for (size_t i = 0; i < projected.size(); ++i)
                    {
                        projected[i] -= scale * equality_coeffs[i];
                    }
                }

                clip_to_bounds(projected, lower_bounds, upper_bounds);
            }

            return projected;
        }

        static double objective_value(const Matrix &q, const std::vector<double> &c, const std::vector<double> &x)
        {
            const std::vector<double> qx = mat_vec(q, x);
            return 0.5 * dot(x, qx) + dot(c, x);
        }

        static double squared_l2_distance(const std::vector<double> &a, const std::vector<double> &b)
        {
            if (a.size() != b.size())
            {
                throw std::invalid_argument("Vector dimensions must match in squared_l2_distance().");
            }

            double sum = 0.0;
            for (size_t i = 0; i < a.size(); ++i)
            {
                const double diff = a[i] - b[i];
                sum += diff * diff;
            }
            return sum;
        }

        static double estimate_lipschitz_constant(const Matrix &q)
        {
            double max_row_sum = 0.0;
            for (size_t i = 0; i < q.numRows(); ++i)
            {
                double row_sum = 0.0;
                for (size_t j = 0; j < q.numCols(); ++j)
                {
                    row_sum += std::abs(q(i, j));
                }
                max_row_sum = std::max(max_row_sum, row_sum);
            }

            if (max_row_sum < 1e-12)
            {
                return 1.0;
            }
            return max_row_sum;
        }

    public:
        QuadraticProgramSolver(
            int max_iter = 5000,
            double tolerance = 1e-6,
            int projection_iters = 30,
            double step_size = 0.0)
            : max_iter(max_iter),
              tolerance(tolerance),
              projection_iters(projection_iters),
              step_size(step_size)
        {
            if (max_iter <= 0)
            {
                throw std::invalid_argument("max_iter must be positive.");
            }
            if (tolerance <= 0)
            {
                throw std::invalid_argument("tolerance must be positive.");
            }
            if (projection_iters <= 0)
            {
                throw std::invalid_argument("projection_iters must be positive.");
            }
            if (step_size < 0)
            {
                throw std::invalid_argument("step_size must be non-negative.");
            }
        }

        QuadraticProgramResult solve(
            const Matrix &q,
            const std::vector<double> &c,
            const std::vector<double> &lower_bounds,
            const std::vector<double> &upper_bounds,
            const std::vector<double> &equality_coeffs = {},
            double equality_value = 0.0,
            const std::vector<double> &initial_solution = {}) const
        {
            if (q.numRows() != q.numCols())
            {
                throw std::invalid_argument("q must be square.");
            }

            const size_t n = q.numRows();
            if (c.size() != n || lower_bounds.size() != n || upper_bounds.size() != n)
            {
                throw std::invalid_argument("q, c, and bounds dimensions must match.");
            }

            if (!equality_coeffs.empty() && equality_coeffs.size() != n)
            {
                throw std::invalid_argument("equality_coeffs size must match variable dimension.");
            }

            for (size_t i = 0; i < n; ++i)
            {
                if (lower_bounds[i] > upper_bounds[i])
                {
                    throw std::invalid_argument("Each lower bound must be <= upper bound.");
                }
            }

            std::vector<double> x(n, 0.0);
            if (!initial_solution.empty())
            {
                if (initial_solution.size() != n)
                {
                    throw std::invalid_argument("initial_solution size must match variable dimension.");
                }
                x = initial_solution;
            }
            else
            {
                for (size_t i = 0; i < n; ++i)
                {
                    const bool lb_finite = std::isfinite(lower_bounds[i]);
                    const bool ub_finite = std::isfinite(upper_bounds[i]);
                    if (lb_finite && ub_finite)
                    {
                        x[i] = 0.5 * (lower_bounds[i] + upper_bounds[i]);
                    }
                    else if (lb_finite)
                    {
                        x[i] = lower_bounds[i];
                    }
                    else if (ub_finite)
                    {
                        x[i] = upper_bounds[i];
                    }
                    else
                    {
                        x[i] = 0.0;
                    }
                }
            }

            x = project_to_feasible_set(x, lower_bounds, upper_bounds, equality_coeffs, equality_value);

            const double lipschitz_constant = estimate_lipschitz_constant(q);
            const double eta = (step_size > 0.0) ? step_size : (1.0 / lipschitz_constant);

            QuadraticProgramResult result;
            result.solution = x;

            for (int iter = 0; iter < max_iter; ++iter)
            {
                const std::vector<double> qx = mat_vec(q, result.solution);
                std::vector<double> gradient(n, 0.0);

                for (size_t i = 0; i < n; ++i)
                {
                    gradient[i] = qx[i] + c[i];
                }

                std::vector<double> next = result.solution;
                for (size_t i = 0; i < n; ++i)
                {
                    next[i] -= eta * gradient[i];
                }

                next = project_to_feasible_set(next, lower_bounds, upper_bounds, equality_coeffs, equality_value);

                const double step_norm_sq = squared_l2_distance(next, result.solution);
                result.solution = next;
                result.iterations = iter + 1;

                if (step_norm_sq <= tolerance * tolerance)
                {
                    result.converged = true;
                    break;
                }
            }

            result.objective = objective_value(q, c, result.solution);
            return result;
        }
    };
} // namespace math
