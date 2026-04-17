#include <gtest/gtest.h>

#include <vector>

#include "math/matrix.hpp"
#include "math/optim_lp.hpp"
#include "math/optim_qp.hpp"

TEST(QuadraticProgramSolver, SolvesBoxConstrainedEqualityProblem)
{
    Matrix q({{2.0, 0.0}, {0.0, 2.0}});
    std::vector<double> c = {-2.0, -4.0};
    std::vector<double> lb = {0.0, 0.0};
    std::vector<double> ub = {1.0, 1.0};
    std::vector<double> aeq = {1.0, 1.0};

    math::QuadraticProgramSolver solver(2000, 1e-8, 40, 0.0);
    const math::QuadraticProgramResult result = solver.solve(q, c, lb, ub, aeq, 1.0);

    EXPECT_TRUE(result.converged);
    ASSERT_EQ(result.solution.size(), 2u);
    EXPECT_NEAR(result.solution[0], 0.0, 1e-3);
    EXPECT_NEAR(result.solution[1], 1.0, 1e-3);
    EXPECT_NEAR(result.objective, -3.0, 1e-3);
}

TEST(LinearProgramSolver, SolvesSimpleMaxProblem)
{
    Matrix a({{1.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}});
    std::vector<double> b = {4.0, 2.0, 3.0};
    std::vector<double> c = {3.0, 2.0};

    math::LinearProgramSolver solver;
    const math::LinearProgramResult result =
        solver.solve(a, b, c, math::LinearProgramSense::Maximize);

    EXPECT_TRUE(result.optimal);
    EXPECT_FALSE(result.unbounded);
    ASSERT_EQ(result.solution.size(), 2u);
    EXPECT_NEAR(result.solution[0], 2.0, 1e-9);
    EXPECT_NEAR(result.solution[1], 2.0, 1e-9);
    EXPECT_NEAR(result.objective, 10.0, 1e-9);
}
