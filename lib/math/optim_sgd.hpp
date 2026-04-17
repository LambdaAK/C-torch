#pragma once

#include "optim_base.hpp"

namespace math
{
    /**
     * @brief Stochastic gradient descent optimizer over mini-batches.
     */
    class SGD
    {
    private:
        double learning_rate; ///< Step size per gradient update.
        int max_iter;         ///< Maximum training iterations.

    public:
        /**
         * @brief Creates SGD optimizer.
         * @param learning_rate Step size.
         * @param max_iter Number of update iterations.
         */
        SGD(double learning_rate, int max_iter) : learning_rate(learning_rate), max_iter(max_iter) {};

        /**
         * @brief Optimizes model parameters using random mini-batches.
         * @param loss_function Sample-level loss generator.
         * @param xTr Training features.
         * @param yTr Training labels.
         * @param initial_theta Initial parameter values.
         * @param batch_size Mini-batch size sampled each iteration.
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

            Differentiator diff;

            for (int i = 0; i < max_iter; i++)
            {
                if (i % 20 == 0)
                {
                    std::cout << "Epoch: " << i << std::endl;
                }
                std::shared_ptr<ASTNode> loss = detail::build_random_batch_loss(loss_function, xTr, yTr, batch_size);

                // compute the gradient of the loss function
                std::unordered_map<std::string, double> gradient = diff.diff(loss, theta);

                // update theta
                for (const auto &pair : gradient)
                {
                    std::string var_name = pair.first;
                    double df_dx = pair.second;
                    theta[var_name] -= learning_rate * df_dx;
                }
            }

            return theta;
        }
    };
} // namespace math
