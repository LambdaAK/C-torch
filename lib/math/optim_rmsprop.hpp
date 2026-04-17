#pragma once

#include "optim_base.hpp"

namespace math
{
    /**
     * @brief RMSProp optimizer for AST-based objectives.
     */
    class RMSProp
    {
    private:
        double learning_rate; ///< Base learning rate.
        int max_iter;         ///< Maximum optimizer iterations.
        double rho;           ///< Exponential decay for squared gradients.
        double epsilon;       ///< Numerical stability constant.

    public:
        /**
         * @brief Creates RMSProp optimizer.
         * @param learning_rate Base learning rate.
         * @param max_iter Number of optimization iterations.
         * @param rho Squared-gradient decay factor.
         * @param epsilon Numerical stability constant.
         */
        RMSProp(double learning_rate, int max_iter, double rho = 0.99, double epsilon = 1e-8)
            : learning_rate(learning_rate), max_iter(max_iter), rho(rho), epsilon(epsilon) {}

        /**
         * @brief Optimizes a single AST objective directly.
         * @param node Objective expression.
         * @param initial_theta Initial parameter values.
         * @return Optimized parameter map.
         */
        std::unordered_map<std::string, double> optimize(
            const std::shared_ptr<ASTNode> &node,
            std::unordered_map<std::string, double> initial_theta)
        {
            std::unordered_map<std::string, double> theta = initial_theta;
            std::unordered_map<std::string, double> avg_sq_grad;
            Differentiator diff;

            for (int i = 0; i < max_iter; i++)
            {
                std::unordered_map<std::string, double> gradient = diff.diff(node, theta);
                for (const auto &pair : gradient)
                {
                    const std::string &var_name = pair.first;
                    const double g = pair.second;
                    avg_sq_grad[var_name] = rho * avg_sq_grad[var_name] + (1.0 - rho) * g * g;
                    theta[var_name] -= learning_rate * g / (std::sqrt(avg_sq_grad[var_name]) + epsilon);
                }
            }

            return theta;
        }

        /**
         * @brief Optimizes supervised loss using random mini-batches.
         * @param loss_function Sample-level loss generator.
         * @param xTr Training features.
         * @param yTr Training labels.
         * @param initial_theta Initial parameter map.
         * @param batch_size Mini-batch size.
         * @return Optimized parameter map.
         */
        std::unordered_map<std::string, double> optimize(
            const std::shared_ptr<LossFunction> &loss_function,
            const Matrix &xTr,
            const Matrix &yTr,
            std::unordered_map<std::string, double> initial_theta,
            int batch_size)
        {
            std::unordered_map<std::string, double> theta = initial_theta;
            std::unordered_map<std::string, double> avg_sq_grad;
            Differentiator diff;

            for (int i = 0; i < max_iter; i++)
            {
                std::shared_ptr<ASTNode> loss = detail::build_random_batch_loss(loss_function, xTr, yTr, batch_size);
                std::unordered_map<std::string, double> gradient = diff.diff(loss, theta);
                for (const auto &pair : gradient)
                {
                    const std::string &var_name = pair.first;
                    const double g = pair.second;
                    avg_sq_grad[var_name] = rho * avg_sq_grad[var_name] + (1.0 - rho) * g * g;
                    theta[var_name] -= learning_rate * g / (std::sqrt(avg_sq_grad[var_name]) + epsilon);
                }
            }

            return theta;
        }
    };
} // namespace math
