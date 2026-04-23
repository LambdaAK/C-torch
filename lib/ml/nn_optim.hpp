#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "math/matrix.hpp"

namespace ml
{
    /**
     * @brief Optimizer families available for neural network parameter updates.
     */
    enum class NNOptimType
    {
        SGD,     ///< Stochastic gradient descent.
        ADAGRAD, ///< AdaGrad adaptive learning-rate optimizer.
        RMSPROP, ///< RMSProp adaptive optimizer.
        ADAM,    ///< Adam optimizer.
        ADAMW    ///< Adam with decoupled weight decay.
    };

    /**
     * @brief Mini-batch optimizer for neural network parameter tensors.
     *
     * The optimizer stores references to parameter and gradient matrices,
     * and applies one update per `step()` call according to `optim_type`.
     */
    class NNOptimizer
    {

    private:
        float lr = 0.0f; ///< Learning rate.
        size_t batch_size = 1; ///< Batch size used to scale gradients.
        NNOptimType optim_type = NNOptimType::SGD; ///< Update rule family.
        float beta1 = 0.9f; ///< First-moment decay (Adam/AdamW).
        float beta2 = 0.999f; ///< Second-moment decay (Adam/AdamW).
        float epsilon = 1e-8f; ///< Numerical stability constant.
        float rho = 0.99f; ///< RMSProp decay factor.
        float weight_decay = 0.0f; ///< Decoupled weight decay (AdamW).
        size_t step_count = 0; ///< Number of optimizer steps applied.
        std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> params; ///< `(parameter, gradient)` pairs.
        std::vector<Matrix> first_moment; ///< First-moment buffers.
        std::vector<Matrix> second_moment; ///< Second-moment buffers.
        std::vector<Matrix> grad_accumulator; ///< Accumulated squared gradients (AdaGrad/RMSProp).

    public:
        /**
         * @brief Constructs an empty optimizer.
         */
        NNOptimizer() = default;

        /**
         * @brief Constructs an optimizer over model parameters.
         * @param p Parameter/gradient matrix pairs to update in-place.
         * @param lr Learning rate.
         * @param batch_size Batch size used for gradient scaling.
         * @param optim_type Optimizer type.
         * @param beta1 First-moment decay (Adam/AdamW).
         * @param beta2 Second-moment decay (Adam/AdamW).
         * @param epsilon Numerical stability constant.
         * @param rho RMSProp decay factor.
         * @param weight_decay Decoupled weight decay coefficient (AdamW).
         */
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

        /**
         * @brief Clears gradient buffers for all tracked parameters.
         */
        void zero_grad();

        /**
         * @brief Returns the batch size used for gradient scaling.
         */
        size_t get_batch_size() const
        {
            return batch_size;
        }

        /**
         * @brief Applies one optimization step to tracked parameters.
         */
        void step();
    };

    /**
     * @brief SGD convenience wrapper around `NNOptimizer`.
     */
    class NN_SGD : public NNOptimizer
    {
    public:
        /**
         * @brief Constructs an empty SGD optimizer.
         */
        NN_SGD() = default;

        /**
         * @brief Constructs SGD optimizer over model parameters.
         * @param p Parameter/gradient pairs.
         * @param lr Learning rate.
         * @param batch_size Batch size used to scale gradients.
         */
        NN_SGD(std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> p, float lr, size_t batch_size)
            : NNOptimizer(std::move(p), lr, batch_size, NNOptimType::SGD) {}
    };

    /**
     * @brief AdaGrad convenience wrapper around `NNOptimizer`.
     */
    class NN_Adagrad : public NNOptimizer
    {
    public:
        /**
         * @brief Constructs an empty AdaGrad optimizer.
         */
        NN_Adagrad() = default;

        /**
         * @brief Constructs AdaGrad optimizer over model parameters.
         * @param p Parameter/gradient pairs.
         * @param lr Learning rate.
         * @param batch_size Batch size used to scale gradients.
         * @param epsilon Numerical stability constant.
         */
        NN_Adagrad(
            std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> p,
            float lr,
            size_t batch_size,
            float epsilon = 1e-8f)
            : NNOptimizer(std::move(p), lr, batch_size, NNOptimType::ADAGRAD, 0.9f, 0.999f, epsilon) {}
    };

    /**
     * @brief RMSProp convenience wrapper around `NNOptimizer`.
     */
    class NN_RMSProp : public NNOptimizer
    {
    public:
        /**
         * @brief Constructs an empty RMSProp optimizer.
         */
        NN_RMSProp() = default;

        /**
         * @brief Constructs RMSProp optimizer over model parameters.
         * @param p Parameter/gradient pairs.
         * @param lr Learning rate.
         * @param batch_size Batch size used to scale gradients.
         * @param rho Exponential decay factor for squared gradients.
         * @param epsilon Numerical stability constant.
         */
        NN_RMSProp(
            std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> p,
            float lr,
            size_t batch_size,
            float rho = 0.99f,
            float epsilon = 1e-8f)
            : NNOptimizer(std::move(p), lr, batch_size, NNOptimType::RMSPROP, 0.9f, 0.999f, epsilon, rho) {}
    };

    /**
     * @brief Adam convenience wrapper around `NNOptimizer`.
     */
    class NN_Adam : public NNOptimizer
    {
    public:
        /**
         * @brief Constructs an empty Adam optimizer.
         */
        NN_Adam() = default;

        /**
         * @brief Constructs Adam optimizer over model parameters.
         * @param p Parameter/gradient pairs.
         * @param lr Learning rate.
         * @param batch_size Batch size used to scale gradients.
         * @param beta1 First-moment decay.
         * @param beta2 Second-moment decay.
         * @param epsilon Numerical stability constant.
         */
        NN_Adam(
            std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> p,
            float lr,
            size_t batch_size,
            float beta1 = 0.9f,
            float beta2 = 0.999f,
            float epsilon = 1e-8f)
            : NNOptimizer(std::move(p), lr, batch_size, NNOptimType::ADAM, beta1, beta2, epsilon) {}
    };

    /**
     * @brief AdamW convenience wrapper around `NNOptimizer`.
     */
    class NN_AdamW : public NNOptimizer
    {
    public:
        /**
         * @brief Constructs an empty AdamW optimizer.
         */
        NN_AdamW() = default;

        /**
         * @brief Constructs AdamW optimizer over model parameters.
         * @param p Parameter/gradient pairs.
         * @param lr Learning rate.
         * @param batch_size Batch size used to scale gradients.
         * @param weight_decay Decoupled weight decay coefficient.
         * @param beta1 First-moment decay.
         * @param beta2 Second-moment decay.
         * @param epsilon Numerical stability constant.
         */
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
