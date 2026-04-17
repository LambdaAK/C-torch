#pragma once

#include "optim_base.hpp"

namespace math
{
    /**
     * @brief Full-batch gradient descent optimizer for AST-defined objectives.
     */
    class GD
    {
    private:
        double learning_rate; ///< Step size per gradient update.
        int max_iter;         ///< Maximum update iterations.

    public:
        /**
         * @brief Creates gradient descent optimizer.
         * @param learning_rate Step size.
         * @param max_iter Maximum iterations.
         */
        GD(double learning_rate, int max_iter) : learning_rate(learning_rate), max_iter(max_iter) {};

        /**
         * @brief Optimizes one scalar variable while keeping others fixed.
         * @param node Objective expression AST.
         * @param var_name Target variable name.
         * @param initial_value Initial variable value.
         * @return Optimized variable value.
         */
        double optimize_single(std::shared_ptr<ASTNode> node, std::string var_name, double initial_value)
        {
            double x = initial_value;
            Differentiator diff;
            for (int i = 0; i < max_iter; i++)
            {
                // compute the derivative at x
                std::shared_ptr<ASTNode> derivative = diff.diff_single(node, var_name, x);
                (void)derivative;
                // update x
                // compute the derivative at the value
                std::shared_ptr<ASTNode> derivative_at_value = diff.diff_single(node, var_name, x);
                // it should be a NumberNode
                std::shared_ptr<NumberNode> cast = std::dynamic_pointer_cast<NumberNode>(derivative_at_value);
                // TODO: make better error handling
                if (cast == nullptr)
                {
                    throw std::runtime_error("The derivative is not a number node.");
                }
                // update x
                x -= learning_rate * cast->getValue();
            }
            return x;
        }

        /**
         * @brief Optimizes all variables in the provided parameter map.
         * @param node Objective expression AST.
         * @param initial_vector Initial variable values keyed by name.
         * @return Optimized variable map.
         */
        std::unordered_map<std::string, double> optimize(std::shared_ptr<ASTNode> node, std::unordered_map<std::string, double> initial_vector)
        {
            std::unordered_map<std::string, double> theta = initial_vector;

            Differentiator diff;

            for (int i = 0; i < max_iter; i++)
            {
                // compute the derivative at the current theta
                std::unordered_map<std::string, double> derivatives = diff.diff(node, theta);

                // update theta
                for (const auto &pair : derivatives)
                {
                    std::string var_name = pair.first;
                    double df_dx = pair.second;
                    theta[var_name] -= learning_rate * df_dx;
                }
                if (i % 20 == 0)
                {
                    std::cout << i << std::endl;
                }
            }

            return theta;
        }
    };
} // namespace math
