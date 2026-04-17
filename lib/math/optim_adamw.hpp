#pragma once

#include "optim_base.hpp"

namespace math
{
    /**
     * @brief AdamW optimizer with decoupled weight decay.
     */
    class AdamW
    {
    private:
        double learning_rate; ///< Base learning rate.
        int max_iter;         ///< Maximum optimizer iterations.
        double beta1;         ///< First-moment decay.
        double beta2;         ///< Second-moment decay.
        double epsilon;       ///< Numerical stability constant.
        double weight_decay;  ///< Decoupled weight decay coefficient.

    public:
        /**
         * @brief Creates AdamW optimizer.
         * @param learning_rate Base learning rate.
         * @param max_iter Number of optimization iterations.
         * @param beta1 First-moment decay.
         * @param beta2 Second-moment decay.
         * @param epsilon Numerical stability constant.
         * @param weight_decay Decoupled weight decay coefficient.
         */
        AdamW(
            double learning_rate,
            int max_iter,
            double beta1 = 0.9,
            double beta2 = 0.999,
            double epsilon = 1e-8,
            double weight_decay = 0.01)
            : learning_rate(learning_rate),
              max_iter(max_iter),
              beta1(beta1),
              beta2(beta2),
              epsilon(epsilon),
              weight_decay(weight_decay) {}

        /**
         * @brief Optimizes a single AST objective directly.
         * @param node Objective expression.
         * @param initial_theta Initial parameter map.
         * @return Optimized parameter map.
         */
        std::unordered_map<std::string, double> optimize(
            const std::shared_ptr<ASTNode> &node,
            std::unordered_map<std::string, double> initial_theta)
        {
            std::unordered_map<std::string, double> theta = initial_theta;
            std::unordered_map<std::string, double> m;
            std::unordered_map<std::string, double> v;
            Differentiator diff;

            for (int t = 1; t <= max_iter; t++)
            {
                std::unordered_map<std::string, double> gradient = diff.diff(node, theta);
                const double one_minus_beta1_t = 1.0 - std::pow(beta1, t);
                const double one_minus_beta2_t = 1.0 - std::pow(beta2, t);

                for (const auto &pair : gradient)
                {
                    const std::string &var_name = pair.first;
                    const double g = pair.second;

                    // Decoupled weight decay.
                    theta[var_name] -= learning_rate * weight_decay * theta[var_name];

                    m[var_name] = beta1 * m[var_name] + (1.0 - beta1) * g;
                    v[var_name] = beta2 * v[var_name] + (1.0 - beta2) * g * g;

                    const double m_hat = m[var_name] / one_minus_beta1_t;
                    const double v_hat = v[var_name] / one_minus_beta2_t;
                    theta[var_name] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
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
            std::unordered_map<std::string, double> m;
            std::unordered_map<std::string, double> v;
            Differentiator diff;

            for (int t = 1; t <= max_iter; t++)
            {
                std::shared_ptr<ASTNode> loss = detail::build_random_batch_loss(loss_function, xTr, yTr, batch_size);
                std::unordered_map<std::string, double> gradient = diff.diff(loss, theta);
                const double one_minus_beta1_t = 1.0 - std::pow(beta1, t);
                const double one_minus_beta2_t = 1.0 - std::pow(beta2, t);

                for (const auto &pair : gradient)
                {
                    const std::string &var_name = pair.first;
                    const double g = pair.second;

                    // Decoupled weight decay.
                    theta[var_name] -= learning_rate * weight_decay * theta[var_name];

                    m[var_name] = beta1 * m[var_name] + (1.0 - beta1) * g;
                    v[var_name] = beta2 * v[var_name] + (1.0 - beta2) * g * g;

                    const double m_hat = m[var_name] / one_minus_beta1_t;
                    const double v_hat = v[var_name] / one_minus_beta2_t;
                    theta[var_name] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
                }
            }

            return theta;
        }
    };
} // namespace math
