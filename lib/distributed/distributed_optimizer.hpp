#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "distributed_training.hpp"
#include "math/optim.hpp"

namespace ctorch::distributed
{
namespace detail
{
inline std::vector<std::string> sorted_parameter_names(const std::unordered_map<std::string, double> &theta)
{
    std::vector<std::string> names;
    names.reserve(theta.size());
    for (const auto &entry : theta)
    {
        names.push_back(entry.first);
    }
    std::sort(names.begin(), names.end());
    return names;
}

inline Matrix pack_gradient_vector(
    const std::unordered_map<std::string, double> &gradient,
    const std::vector<std::string> &names)
{
    Matrix packed(1, names.size());
    for (std::size_t index = 0; index < names.size(); ++index)
    {
        const auto it = gradient.find(names[index]);
        if (it == gradient.end())
        {
            throw std::runtime_error("Distributed optimizer gradient is missing variable: " + names[index]);
        }
        packed(0, index) = it->second;
    }
    return packed;
}

inline std::unordered_map<std::string, double> unpack_gradient_vector(
    const Matrix &packed,
    const std::vector<std::string> &names)
{
    if (packed.numRows() != 1 || packed.numCols() != names.size())
    {
        throw std::runtime_error("Packed gradient vector shape mismatch.");
    }

    std::unordered_map<std::string, double> gradient;
    for (std::size_t index = 0; index < names.size(); ++index)
    {
        gradient[names[index]] = packed(0, index);
    }
    return gradient;
}
} // namespace detail

/**
 * @brief Synchronous data-parallel optimizer for AST-based supervised models.
 *
 * The optimizer shards the training set evenly across workers, computes the
 * average local gradient on each rank, performs an AllReduce, and then applies
 * the same update rule on every worker. This keeps all parameter replicas in
 * lockstep.
 */
class DistributedOptimizer
{
private:
    ProcessGroup &group_;
    math::OptimParams optim_params_;

    static void apply_update(
        math::OptimType optim_type,
        double learning_rate,
        double beta1,
        double beta2,
        double epsilon,
        double rho,
        double weight_decay,
        std::size_t step_count,
        std::unordered_map<std::string, double> &theta,
        std::unordered_map<std::string, double> &first_moment,
        std::unordered_map<std::string, double> &second_moment,
        std::unordered_map<std::string, double> &grad_accumulator,
        const std::unordered_map<std::string, double> &gradient)
    {
        const double one_minus_beta1_t = 1.0 - std::pow(beta1, static_cast<double>(step_count));
        const double one_minus_beta2_t = 1.0 - std::pow(beta2, static_cast<double>(step_count));

        for (const auto &entry : gradient)
        {
            const std::string &var_name = entry.first;
            const double g = entry.second;

            if (optim_type == math::OptimType::GD)
            {
                theta[var_name] -= learning_rate * g;
                continue;
            }

            if (optim_type == math::OptimType::SGD)
            {
                theta[var_name] -= learning_rate * g;
                continue;
            }

            if (optim_type == math::OptimType::ADAGRAD)
            {
                grad_accumulator[var_name] += g * g;
                theta[var_name] -= learning_rate * g / (std::sqrt(grad_accumulator[var_name]) + epsilon);
                continue;
            }

            if (optim_type == math::OptimType::RMSPROP)
            {
                second_moment[var_name] = rho * second_moment[var_name] + (1.0 - rho) * g * g;
                theta[var_name] -= learning_rate * g / (std::sqrt(second_moment[var_name]) + epsilon);
                continue;
            }

            if (optim_type == math::OptimType::ADAM || optim_type == math::OptimType::ADAMW)
            {
                if (optim_type == math::OptimType::ADAMW)
                {
                    theta[var_name] -= learning_rate * weight_decay * theta[var_name];
                }

                first_moment[var_name] = beta1 * first_moment[var_name] + (1.0 - beta1) * g;
                second_moment[var_name] = beta2 * second_moment[var_name] + (1.0 - beta2) * g * g;
                const double m_hat = first_moment[var_name] / one_minus_beta1_t;
                const double v_hat = second_moment[var_name] / one_minus_beta2_t;
                theta[var_name] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
                continue;
            }

            throw std::runtime_error("Unknown optimizer type in distributed update.");
        }
    }

public:
    DistributedOptimizer(ProcessGroup &group, math::OptimParams optim_params)
        : group_(group), optim_params_(std::move(optim_params))
    {
    }

    std::unordered_map<std::string, double> optimize(
        const std::shared_ptr<LossFunction> &loss_function,
        const Matrix &xTr,
        const Matrix &yTr,
        std::unordered_map<std::string, double> initial_theta) const
    {
        if (!loss_function)
        {
            throw std::invalid_argument("loss_function must not be null.");
        }

        if (group_.world_size() == 1)
        {
            math::Optimizer optimizer(optim_params_);
            return optimizer.optimize(loss_function, xTr, yTr, std::move(initial_theta));
        }

        if (optim_params_.get_optim_type() != math::OptimType::GD)
        {
            const int batch_size = optim_params_.get_batch_size();
            if (batch_size <= 0)
            {
                throw std::invalid_argument("batch_size must be positive.");
            }
            if (static_cast<std::size_t>(batch_size) % static_cast<std::size_t>(group_.world_size()) != 0)
            {
                throw std::invalid_argument("batch_size must be divisible by world_size in distributed mode.");
            }
        }

        const auto [local_xTr, local_yTr] = equal_row_shard(xTr, yTr, group_.rank(), group_.world_size());
        if (local_xTr.numRows() == 0)
        {
            throw std::invalid_argument("Distributed shard is empty.");
        }

        std::unordered_map<std::string, double> theta = std::move(initial_theta);
        const std::vector<std::string> parameter_names = detail::sorted_parameter_names(theta);
        if (parameter_names.empty())
        {
            return theta;
        }

        std::srand(static_cast<unsigned int>(group_.rank() + 1));

        std::unordered_map<std::string, double> first_moment;
        std::unordered_map<std::string, double> second_moment;
        std::unordered_map<std::string, double> grad_accumulator;

        math::Differentiator diff;
        for (int iter = 0; iter < optim_params_.get_max_iter(); ++iter)
        {
            std::shared_ptr<math::ASTNode> local_loss;
            if (optim_params_.get_optim_type() == math::OptimType::GD)
            {
                local_loss = math::detail::build_full_batch_loss(loss_function, local_xTr, local_yTr);
            }
            else
            {
                const int local_batch_size = optim_params_.get_batch_size() / group_.world_size();
                if (local_batch_size == 0)
                {
                    throw std::invalid_argument("Distributed local batch size must be positive.");
                }
                local_loss = math::detail::build_random_batch_loss(loss_function, local_xTr, local_yTr, local_batch_size);
            }

            std::unordered_map<std::string, double> local_gradient = diff.diff(local_loss, theta);
            Matrix packed_gradient = detail::pack_gradient_vector(local_gradient, parameter_names);
            group_.allreduce_sum(packed_gradient);
            packed_gradient = packed_gradient * (1.0 / static_cast<double>(group_.world_size()));

            const std::unordered_map<std::string, double> gradient =
                detail::unpack_gradient_vector(packed_gradient, parameter_names);

            apply_update(
                optim_params_.get_optim_type(),
                optim_params_.get_learning_rate(),
                optim_params_.get_beta1(),
                optim_params_.get_beta2(),
                optim_params_.get_epsilon(),
                optim_params_.get_rho(),
                optim_params_.get_weight_decay(),
                static_cast<std::size_t>(iter + 1),
                theta,
                first_moment,
                second_moment,
                grad_accumulator,
                gradient);

            if (group_.rank() == 0 && iter % 20 == 0)
            {
                std::cout << "Distributed epoch: " << iter << std::endl;
            }
        }

        return theta;
    }
};
} // namespace ctorch::distributed
