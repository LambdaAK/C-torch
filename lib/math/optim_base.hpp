#pragma once

#include "ast.hpp"
#include "lossfunction.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace math
{

    /**
     * @brief Optimizer families supported by the optimization stack.
     */
    enum class OptimType
    {
        GD,      ///< Full-batch gradient descent.
        SGD,     ///< Stochastic gradient descent.
        ADAGRAD, ///< AdaGrad adaptive optimizer.
        RMSPROP, ///< RMSProp adaptive optimizer.
        ADAM,    ///< Adam adaptive optimizer.
        ADAMW    ///< AdamW (Adam + decoupled weight decay).
    };

    /**
     * @brief Immutable optimization configuration bundle.
     */
    class OptimParams
    {
    private:
        OptimType optim_type;  ///< Selected optimizer family.
        double learning_rate;  ///< Optimizer learning rate.
        int max_iter;          ///< Maximum iterations.
        Matrix xTr;            ///< Optional training feature cache.
        Matrix yTr;            ///< Optional training label cache.
        int batch_size;        ///< Mini-batch size for stochastic optimizers.
        double beta1;          ///< First-moment decay for Adam-like optimizers.
        double beta2;          ///< Second-moment decay for Adam-like optimizers.
        double epsilon;        ///< Numerical stability constant.
        double rho;            ///< RMSProp smoothing factor.
        double weight_decay;   ///< Decoupled weight decay coefficient.

    public:
        /**
         * @brief Constructs optimizer configuration with optional hyperparameters.
         * @param optim_type Optimizer family.
         * @param learning_rate Step size.
         * @param max_iter Maximum iterations.
         * @param xTr Optional training features.
         * @param yTr Optional training labels.
         * @param batch_size Mini-batch size.
         * @param beta1 First-moment decay.
         * @param beta2 Second-moment decay.
         * @param epsilon Numerical stability constant.
         * @param rho RMSProp smoothing factor.
         * @param weight_decay Decoupled weight decay.
         */
        OptimParams(
            OptimType optim_type = OptimType::GD,
            double learning_rate = 0.001,
            int max_iter = 1000,
            Matrix xTr = {{}},
            Matrix yTr = {},
            int batch_size = 1,
            double beta1 = 0.9,
            double beta2 = 0.999,
            double epsilon = 1e-8,
            double rho = 0.99,
            double weight_decay = 0.0)
            : optim_type(optim_type),
              learning_rate(learning_rate),
              max_iter(max_iter),
              xTr(xTr),
              yTr(yTr),
              batch_size(batch_size),
              beta1(beta1),
              beta2(beta2),
              epsilon(epsilon),
              rho(rho),
              weight_decay(weight_decay) {}

        /**
         * @brief Returns optimizer family.
         * @return `OptimType` enum.
         */
        OptimType get_optim_type() const
        {
            return optim_type;
        }

        /**
         * @brief Returns configured learning rate.
         * @return Learning rate value.
         */
        double get_learning_rate() const
        {
            return learning_rate;
        }

        /**
         * @brief Returns cached training features.
         * @return Training feature matrix.
         */
        Matrix get_xTr() const
        {
            return xTr;
        }

        /**
         * @brief Returns cached training labels.
         * @return Training label matrix.
         */
        Matrix get_yTr() const
        {
            return yTr;
        }

        /**
         * @brief Returns maximum optimizer iterations.
         * @return Iteration count.
         */
        int get_max_iter() const
        {
            return max_iter;
        }

        /**
         * @brief Returns configured batch size.
         * @return Batch size.
         */
        int get_batch_size() const
        {
            return batch_size;
        }

        /**
         * @brief Returns beta1 hyperparameter.
         * @return First-moment decay.
         */
        double get_beta1() const
        {
            return beta1;
        }

        /**
         * @brief Returns beta2 hyperparameter.
         * @return Second-moment decay.
         */
        double get_beta2() const
        {
            return beta2;
        }

        /**
         * @brief Returns epsilon hyperparameter.
         * @return Numerical stability constant.
         */
        double get_epsilon() const
        {
            return epsilon;
        }

        /**
         * @brief Returns rho hyperparameter.
         * @return RMSProp smoothing factor.
         */
        double get_rho() const
        {
            return rho;
        }

        /**
         * @brief Returns decoupled weight decay coefficient.
         * @return Weight decay value.
         */
        double get_weight_decay() const
        {
            return weight_decay;
        }
    };

    namespace detail
    {
        /**
         * @brief Validates supervised data layout constraints.
         * @param xTr Training features.
         * @param yTr Training labels.
         */
        inline void validate_supervised_data(const Matrix &xTr, const Matrix &yTr)
        {
            if (xTr.numRows() == 0 || xTr.numCols() == 0)
            {
                throw std::invalid_argument("xTr must be non-empty.");
            }
            if (yTr.numRows() != 1)
            {
                throw std::invalid_argument("yTr must be a row vector (1 x N).");
            }
            if (xTr.numRows() != yTr.numCols())
            {
                throw std::invalid_argument("Number of samples in xTr must match number of labels in yTr.");
            }
        }

        /**
         * @brief Normalizes batch size to a minimum of 1.
         * @param batch_size Requested batch size.
         * @return Clamped batch size.
         */
        inline int normalize_batch_size(int batch_size)
        {
            return std::max(1, batch_size);
        }

        /**
         * @brief Builds full-batch loss expression from a loss function object.
         * @param loss_function Loss implementation.
         * @param xTr Training features.
         * @param yTr Training labels.
         * @return AST representing mean sample loss plus regularizer.
         */
        inline std::shared_ptr<ASTNode> build_full_batch_loss(
            const std::shared_ptr<LossFunction> &loss_function,
            const Matrix &xTr,
            const Matrix &yTr)
        {
            validate_supervised_data(xTr, yTr);

            std::shared_ptr<ASTNode> loss = Num(0);
            std::vector<Matrix> xTr_rows = xTr.rowsAsMatrices();

            for (size_t i = 0; i < xTr.numRows(); i++)
            {
                Matrix x = xTr_rows[i];
                int y = static_cast<int>(yTr.at(0, i));
                std::shared_ptr<ASTNode> sample_loss = loss_function->sample_loss(x, y);
                loss = loss + sample_loss;
            }

            loss = loss / Num(static_cast<double>(xTr.numRows()));
            loss = loss + loss_function->regularizer();
            return loss;
        }

        /**
         * @brief Builds random mini-batch loss expression.
         * @param loss_function Loss implementation.
         * @param xTr Training features.
         * @param yTr Training labels.
         * @param batch_size Requested mini-batch size.
         * @return AST representing sampled mean loss plus regularizer.
         */
        inline std::shared_ptr<ASTNode> build_random_batch_loss(
            const std::shared_ptr<LossFunction> &loss_function,
            const Matrix &xTr,
            const Matrix &yTr,
            int batch_size)
        {
            validate_supervised_data(xTr, yTr);
            const int normalized_batch_size = normalize_batch_size(batch_size);

            std::shared_ptr<ASTNode> loss = Num(0);
            std::vector<Matrix> xTr_rows = xTr.rowsAsMatrices();
            const int num_rows = static_cast<int>(xTr.numRows());

            for (int j = 0; j < normalized_batch_size; j++)
            {
                int index = std::rand() % num_rows;
                Matrix x = xTr_rows[index];
                int y = static_cast<int>(yTr.at(0, index));
                std::shared_ptr<ASTNode> sample_loss = loss_function->sample_loss(x, y);
                loss = loss + sample_loss;
            }

            loss = loss / Num(normalized_batch_size);
            loss = loss + loss_function->regularizer();
            return loss;
        }
    } // namespace detail

    /**
     * @brief Numerical differentiator for AST-based objective functions.
     *
     * Uses finite-difference approximations to estimate gradients.
     */
    class Differentiator
    {

        /*
            We know that

            f'(x) = lim_{h -> 0} (f(x + h) - f(x)) / h

            Therefore, for small h, we can approximate the derivative of f at x as

            f'(x) = (f(x + h) - f(x)) / h
        */

    private:
        double h = 1e-5; ///< Finite-difference step size.

    public:
        /**
         * @brief Differentiates expression with respect to one variable.
         * @param node Expression AST.
         * @param var_name Variable name to differentiate by.
         * @param value Evaluation point for variable.
         * @return Simplified AST approximating derivative value at `value`.
         */
        std::shared_ptr<ASTNode> diff_single(std::shared_ptr<ASTNode> node, std::string var_name, double value)
        {

            // compute f(x + h)
            std::shared_ptr<ASTNode> f_x_plus_h = node->substitute(var_name, Num(value + h));

            // compute f(x)
            std::shared_ptr<ASTNode> f_x = node->substitute(var_name, Num(value));

            // compute f(x + h) - f(x)
            std::shared_ptr<ASTNode> diff = f_x_plus_h - f_x;

            // compute (f(x + h) - f(x)) / h
            std::shared_ptr<ASTNode> derivative_at_value = diff / Num(h);

            return derivative_at_value->simplify();
        }

        /**
         * @brief Computes gradient map with respect to all provided variables.
         * @param node Expression AST.
         * @param values Variable assignment map for evaluation.
         * @return Partial derivatives keyed by variable name.
         */
        std::unordered_map<std::string, double> diff(std::shared_ptr<ASTNode> node, std::unordered_map<std::string, double> values)
        {
            // compute the derivative with respect to each variable

            std::unordered_map<std::string, double> derivatives;

            for (const auto &pair : values)
            {
                std::string var_name = pair.first;
                // compute the partial derivative
                // compute f(x + h)
                std::shared_ptr<ASTNode> f_x_plus_h = node->substitute(var_name, Num(values[var_name] + h));
                // compute f(x - h)
                std::shared_ptr<ASTNode> f_x_minus_h = node->substitute(var_name, Num(values[var_name] - h));
                // compute f(x + h) - f(x)
                std::shared_ptr<ASTNode> diff = f_x_plus_h - f_x_minus_h;
                // compute (f(x + h) - f(x - h)) / (2h)
                std::shared_ptr<ASTNode> acc = diff / (Num(2 * h));

                // evaluate the derivative at the values provided

                for (const auto &pair : values)
                {
                    acc = acc->substitute(pair.first, Num(pair.second));
                }

                acc = acc->simplify();

                // this should now be a NumberNode

                std::shared_ptr<NumberNode> cast = std::dynamic_pointer_cast<NumberNode>(acc);

                // if cast == nullptr, throw an error and show what the missing variables are

                if (cast == nullptr)
                {
                    std::set<std::string> vars = acc->variables();
                    std::string missing_vars = "";
                    for (const auto &var : vars)
                    {
                        missing_vars += var + " ";
                    }
                    throw std::runtime_error("The derivative is not a number node in diff(). Missing variables: " + missing_vars);
                }

                derivatives[var_name] = cast->getValue();
            }
            return derivatives;
        }
    };

} // namespace math
