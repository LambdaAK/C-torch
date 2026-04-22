#include "nn_optim.hpp"
#include "math/parallel.hpp"

#include <cmath>
#include <exception>
#include <stdexcept>
#include <mutex>
#include <utility>

namespace ml
{
    NNOptimizer::NNOptimizer(
        std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> p,
        float lr,
        size_t batch_size,
        NNOptimType optim_type,
        float beta1,
        float beta2,
        float epsilon,
        float rho,
        float weight_decay)
        : lr(lr),
          batch_size(batch_size),
          optim_type(optim_type),
          beta1(beta1),
          beta2(beta2),
          epsilon(epsilon),
          rho(rho),
          weight_decay(weight_decay),
          step_count(0),
          params(std::move(p))
    {
        first_moment.reserve(params.size());
        second_moment.reserve(params.size());
        grad_accumulator.reserve(params.size());

        for (const auto &p : params)
        {
            if (!p.first || !p.second)
            {
                throw std::invalid_argument("NNOptimizer received null parameter or gradient pointer.");
            }

            const size_t rows = p.first->numRows();
            const size_t cols = p.first->numCols();

            if (p.second->numRows() != rows || p.second->numCols() != cols)
            {
                throw std::invalid_argument("NNOptimizer parameter and gradient dimensions do not match.");
            }

            first_moment.emplace_back(rows, cols);
            second_moment.emplace_back(rows, cols);
            grad_accumulator.emplace_back(rows, cols);
        }
    }

    void NNOptimizer::zero_grad()
    {
        ctorch::parallel::parallel_for_items(
            params.size(),
            1,
            [&](std::size_t begin, std::size_t end)
            {
                for (std::size_t idx = begin; idx < end; ++idx)
                {
                    (*params[idx].second) = 0.0 * (*params[idx].second); // zero the gradients
                }
            },
            2);
    }

    void NNOptimizer::step()
    {
        if (batch_size == 0)
        {
            throw std::invalid_argument("NNOptimizer::step called with batch_size = 0.");
        }

        ++step_count;
        const float inv_batch_size = 1.0f / static_cast<float>(batch_size);
        const float one_minus_beta1_t = 1.0f - std::pow(beta1, static_cast<float>(step_count));
        const float one_minus_beta2_t = 1.0f - std::pow(beta2, static_cast<float>(step_count));

        ctorch::parallel::parallel_for_items(
            params.size(),
            1,
            [&](std::size_t begin, std::size_t end)
            {
                for (std::size_t idx = begin; idx < end; ++idx)
                {
                    Matrix &param = *params[idx].first;
                    const Matrix grad = inv_batch_size * (*params[idx].second);
                    Matrix &m = first_moment[idx];
                    Matrix &v = second_moment[idx];
                    Matrix &acc = grad_accumulator[idx];

                    for (size_t i = 0; i < param.numRows(); ++i)
                    {
                        for (size_t j = 0; j < param.numCols(); ++j)
                        {
                            const float g = static_cast<float>(grad(i, j));

                            if (optim_type == NNOptimType::SGD)
                            {
                                param(i, j) -= lr * g;
                                continue;
                            }

                            if (optim_type == NNOptimType::ADAGRAD)
                            {
                                acc(i, j) = acc(i, j) + g * g;
                                param(i, j) -= lr * g / (std::sqrt(acc(i, j)) + epsilon);
                                continue;
                            }

                            if (optim_type == NNOptimType::RMSPROP)
                            {
                                v(i, j) = rho * v(i, j) + (1.0f - rho) * g * g;
                                param(i, j) -= lr * g / (std::sqrt(v(i, j)) + epsilon);
                                continue;
                            }

                            if (optim_type == NNOptimType::ADAM || optim_type == NNOptimType::ADAMW)
                            {
                                if (optim_type == NNOptimType::ADAMW)
                                {
                                    // Decoupled weight decay.
                                    param(i, j) -= lr * weight_decay * static_cast<float>(param(i, j));
                                }

                                m(i, j) = beta1 * m(i, j) + (1.0f - beta1) * g;
                                v(i, j) = beta2 * v(i, j) + (1.0f - beta2) * g * g;
                                const float m_hat = m(i, j) / one_minus_beta1_t;
                                const float v_hat = v(i, j) / one_minus_beta2_t;
                                param(i, j) -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
                                continue;
                            }

                            throw std::runtime_error("Unknown NN optimizer type.");
                        }
                    }
                }
            },
            2);
    }
} // namespace ml
