#pragma once

#include "optim_base.hpp"

namespace math
{
    class Adagrad
    {
    private:
        double learning_rate;
        int max_iter;
        double epsilon;

    public:
        Adagrad(double learning_rate, int max_iter, double epsilon = 1e-8)
            : learning_rate(learning_rate), max_iter(max_iter), epsilon(epsilon) {}

        std::unordered_map<std::string, double> optimize(
            const std::shared_ptr<ASTNode> &node,
            std::unordered_map<std::string, double> initial_theta)
        {
            std::unordered_map<std::string, double> theta = initial_theta;
            std::unordered_map<std::string, double> accumulator;
            Differentiator diff;

            for (int i = 0; i < max_iter; i++)
            {
                std::unordered_map<std::string, double> gradient = diff.diff(node, theta);
                for (const auto &pair : gradient)
                {
                    const std::string &var_name = pair.first;
                    const double g = pair.second;
                    accumulator[var_name] += g * g;
                    theta[var_name] -= learning_rate * g / (std::sqrt(accumulator[var_name]) + epsilon);
                }
            }

            return theta;
        }

        std::unordered_map<std::string, double> optimize(
            const std::shared_ptr<LossFunction> &loss_function,
            const Matrix &xTr,
            const Matrix &yTr,
            std::unordered_map<std::string, double> initial_theta,
            int batch_size)
        {
            std::unordered_map<std::string, double> theta = initial_theta;
            std::unordered_map<std::string, double> accumulator;
            Differentiator diff;

            for (int i = 0; i < max_iter; i++)
            {
                std::shared_ptr<ASTNode> loss = detail::build_random_batch_loss(loss_function, xTr, yTr, batch_size);
                std::unordered_map<std::string, double> gradient = diff.diff(loss, theta);
                for (const auto &pair : gradient)
                {
                    const std::string &var_name = pair.first;
                    const double g = pair.second;
                    accumulator[var_name] += g * g;
                    theta[var_name] -= learning_rate * g / (std::sqrt(accumulator[var_name]) + epsilon);
                }
            }

            return theta;
        }
    };
} // namespace math

