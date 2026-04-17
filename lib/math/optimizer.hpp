#pragma once

#include "optim_adagrad.hpp"
#include "optim_adam.hpp"
#include "optim_adamw.hpp"
#include "optim_gd.hpp"
#include "optim_rmsprop.hpp"
#include "optim_sgd.hpp"

namespace math
{
    /**
     * @brief High-level optimizer dispatcher.
     *
     * Routes optimization requests to concrete optimizer implementations
     * based on `OptimParams::get_optim_type()`.
     */
    class Optimizer
    {
    private:
        OptimParams optim_params; ///< Optimizer configuration.

    public:
        /**
         * @brief Creates dispatcher with fixed optimizer configuration.
         * @param optim_params Optimizer family and hyperparameters.
         */
        Optimizer(OptimParams optim_params) : optim_params(optim_params) {}

        /**
         * @brief Optimizes parameters for a supervised loss function.
         * @param loss_function Sample-level loss generator.
         * @param xTr Training features.
         * @param yTr Training labels.
         * @param initial_theta Initial parameter map.
         * @return Optimized parameter map.
         */
        std::unordered_map<std::string, double> optimize(
            const std::shared_ptr<LossFunction> &loss_function,
            const Matrix &xTr,
            const Matrix &yTr,
            std::unordered_map<std::string, double> initial_theta)
        {
            if (optim_params.get_optim_type() == OptimType::GD)
            {
                GD gd(optim_params.get_learning_rate(), optim_params.get_max_iter());
                std::shared_ptr<ASTNode> loss = detail::build_full_batch_loss(loss_function, xTr, yTr);
                std::unordered_map<std::string, double> theta = gd.optimize(loss, initial_theta);
                return theta;
            }

            if (optim_params.get_optim_type() == OptimType::SGD)
            {
                SGD sgd(optim_params.get_learning_rate(), optim_params.get_max_iter());
                std::unordered_map<std::string, double> theta =
                    sgd.optimize(loss_function, xTr, yTr, initial_theta, optim_params.get_batch_size());
                return theta;
            }

            if (optim_params.get_optim_type() == OptimType::ADAGRAD)
            {
                Adagrad adagrad(optim_params.get_learning_rate(), optim_params.get_max_iter(), optim_params.get_epsilon());
                std::unordered_map<std::string, double> theta =
                    adagrad.optimize(loss_function, xTr, yTr, initial_theta, optim_params.get_batch_size());
                return theta;
            }

            if (optim_params.get_optim_type() == OptimType::RMSPROP)
            {
                RMSProp rmsprop(
                    optim_params.get_learning_rate(),
                    optim_params.get_max_iter(),
                    optim_params.get_rho(),
                    optim_params.get_epsilon());
                std::unordered_map<std::string, double> theta =
                    rmsprop.optimize(loss_function, xTr, yTr, initial_theta, optim_params.get_batch_size());
                return theta;
            }

            if (optim_params.get_optim_type() == OptimType::ADAM)
            {
                Adam adam(
                    optim_params.get_learning_rate(),
                    optim_params.get_max_iter(),
                    optim_params.get_beta1(),
                    optim_params.get_beta2(),
                    optim_params.get_epsilon());
                std::unordered_map<std::string, double> theta =
                    adam.optimize(loss_function, xTr, yTr, initial_theta, optim_params.get_batch_size());
                return theta;
            }

            if (optim_params.get_optim_type() == OptimType::ADAMW)
            {
                AdamW adamw(
                    optim_params.get_learning_rate(),
                    optim_params.get_max_iter(),
                    optim_params.get_beta1(),
                    optim_params.get_beta2(),
                    optim_params.get_epsilon(),
                    optim_params.get_weight_decay());
                std::unordered_map<std::string, double> theta =
                    adamw.optimize(loss_function, xTr, yTr, initial_theta, optim_params.get_batch_size());
                return theta;
            }

            throw std::runtime_error("Unknown optimizer type");
        }
    };
} // namespace math
