#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "math/matrix.hpp"

namespace ml
{
    enum class NNOptimType
    {
        SGD,
        ADAGRAD,
        RMSPROP,
        ADAM,
        ADAMW
    };

    class NNOptimizer
    {

    private:
        float lr = 0.0f;
        size_t batch_size = 1;
        NNOptimType optim_type = NNOptimType::SGD;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float epsilon = 1e-8f;
        float rho = 0.99f;
        float weight_decay = 0.0f;
        size_t step_count = 0;
        std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> params;
        std::vector<Matrix> first_moment;
        std::vector<Matrix> second_moment;
        std::vector<Matrix> grad_accumulator;

    public:
        NNOptimizer() = default;
        NNOptimizer(
            std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> p,
            float lr,
            size_t batch_size,
            NNOptimType optim_type = NNOptimType::SGD,
            float beta1 = 0.9f,
            float beta2 = 0.999f,
            float epsilon = 1e-8f,
            float rho = 0.99f,
            float weight_decay = 0.0f);
        void zero_grad();
        void step();
    };

    class NN_SGD : public NNOptimizer
    {
    public:
        NN_SGD() = default;
        NN_SGD(std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> p, float lr, size_t batch_size)
            : NNOptimizer(std::move(p), lr, batch_size, NNOptimType::SGD) {}
    };

    class NN_Adagrad : public NNOptimizer
    {
    public:
        NN_Adagrad() = default;
        NN_Adagrad(
            std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> p,
            float lr,
            size_t batch_size,
            float epsilon = 1e-8f)
            : NNOptimizer(std::move(p), lr, batch_size, NNOptimType::ADAGRAD, 0.9f, 0.999f, epsilon) {}
    };

    class NN_RMSProp : public NNOptimizer
    {
    public:
        NN_RMSProp() = default;
        NN_RMSProp(
            std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> p,
            float lr,
            size_t batch_size,
            float rho = 0.99f,
            float epsilon = 1e-8f)
            : NNOptimizer(std::move(p), lr, batch_size, NNOptimType::RMSPROP, 0.9f, 0.999f, epsilon, rho) {}
    };

    class NN_Adam : public NNOptimizer
    {
    public:
        NN_Adam() = default;
        NN_Adam(
            std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> p,
            float lr,
            size_t batch_size,
            float beta1 = 0.9f,
            float beta2 = 0.999f,
            float epsilon = 1e-8f)
            : NNOptimizer(std::move(p), lr, batch_size, NNOptimType::ADAM, beta1, beta2, epsilon) {}
    };

    class NN_AdamW : public NNOptimizer
    {
    public:
        NN_AdamW() = default;
        NN_AdamW(
            std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> p,
            float lr,
            size_t batch_size,
            float weight_decay = 0.01f,
            float beta1 = 0.9f,
            float beta2 = 0.999f,
            float epsilon = 1e-8f)
            : NNOptimizer(std::move(p), lr, batch_size, NNOptimType::ADAMW, beta1, beta2, epsilon, 0.99f, weight_decay) {}
    };
} // namespace ml

