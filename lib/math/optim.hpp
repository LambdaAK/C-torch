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

    enum class OptimType
    {
        GD,
        SGD,
        ADAGRAD,
        RMSPROP,
        ADAM,
        ADAMW
    };

    class OptimParams
    {
    private:
        OptimType optim_type;
        double learning_rate;
        int max_iter;
        Matrix xTr;
        Matrix yTr;
        int batch_size;
        double beta1;
        double beta2;
        double epsilon;
        double rho;
        double weight_decay;

    public:
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

        OptimType get_optim_type() const
        {
            return optim_type;
        }

        double get_learning_rate() const
        {
            return learning_rate;
        }

        Matrix get_xTr() const
        {
            return xTr;
        }

        Matrix get_yTr() const
        {
            return yTr;
        }

        int get_max_iter() const
        {
            return max_iter;
        }

        int get_batch_size() const
        {
            return batch_size;
        }

        double get_beta1() const
        {
            return beta1;
        }

        double get_beta2() const
        {
            return beta2;
        }

        double get_epsilon() const
        {
            return epsilon;
        }

        double get_rho() const
        {
            return rho;
        }

        double get_weight_decay() const
        {
            return weight_decay;
        }
    };

    namespace detail
    {
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

        inline int normalize_batch_size(int batch_size)
        {
            return std::max(1, batch_size);
        }

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

    class Differentiator
    {

        /*
            We know that

            f'(x) = lim_{h -> 0} (f(x + h) - f(x)) / h

            Therefore, for small h, we can approximate the derivative of f at x as

            f'(x) = (f(x + h) - f(x)) / h
        */

    private:
        double h = 1e-5;

    public:
        /**
         * Differentiate the function with respect to var_name and return the derivative evaluated at the value
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
         * Differentiate with respect to each variable in values, and return the derivative evaluated the values provided
         * [values] should map every variable in node to a value, otherwise, an error will be thrown
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

    class GD
    {
    private:
        double learning_rate;
        int max_iter;

    public:
        GD(double learning_rate, int max_iter) : learning_rate(learning_rate), max_iter(max_iter) {};

        /**
         * Optimize the function represented by the AST node with respect to var_name
         */
        double optimize_single(std::shared_ptr<ASTNode> node, std::string var_name, double initial_value)
        {
            double x = initial_value;
            Differentiator diff;
            for (int i = 0; i < max_iter; i++)
            {
                // compute the derivative at x
                std::shared_ptr<ASTNode> derivative = diff.diff_single(node, var_name, x);
                // update x
                // compute the derivative at the value
                std::shared_ptr<ASTNode> derivative_at_value = diff.diff_single(node, var_name, x);
                // it should be a NumberNode
                std::shared_ptr<NumberNode> cast = std::dynamic_pointer_cast<NumberNode>(derivative_at_value);
                // TODO: make better error handling
                if (cast == nullptr)
                {
                    throw std::runtime_error("The derivative is not a number node.");
                }
                // update x
                x -= learning_rate * cast->getValue();
            }
            return x;
        }

        /**
         * Optimize the funciton with respect to all variables
         * initial_vector maps from the variable names to the initial values
         */
        std::unordered_map<std::string, double> optimize(std::shared_ptr<ASTNode> node, std::unordered_map<std::string, double> initial_vector)
        {
            std::unordered_map<std::string, double> theta = initial_vector;

            Differentiator diff;

            for (int i = 0; i < max_iter; i++)
            {
                // compute the derivative at the current theta
                std::unordered_map<std::string, double> derivatives = diff.diff(node, theta);

                // update theta
                for (const auto &pair : derivatives)
                {
                    std::string var_name = pair.first;
                    double df_dx = pair.second;
                    theta[var_name] -= learning_rate * df_dx;
                }
                if (i % 20 == 0)
                {
                    std::cout << i << std::endl;
                }
            }

            return theta;
        }
    };

    class SGD
    {
    private:
        double learning_rate;
        int max_iter;

    public:
        SGD(double learning_rate, int max_iter) : learning_rate(learning_rate), max_iter(max_iter) {};

        /*
            Optimize method
            Takes in
                Loss function generator
                xTr
                yTr
                initial theta
                batch size
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

    class RMSProp
    {
    private:
        double learning_rate;
        int max_iter;
        double rho;
        double epsilon;

    public:
        RMSProp(double learning_rate, int max_iter, double rho = 0.99, double epsilon = 1e-8)
            : learning_rate(learning_rate), max_iter(max_iter), rho(rho), epsilon(epsilon) {}

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

    class Adam
    {
    private:
        double learning_rate;
        int max_iter;
        double beta1;
        double beta2;
        double epsilon;

    public:
        Adam(double learning_rate, int max_iter, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
            : learning_rate(learning_rate), max_iter(max_iter), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

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
                    m[var_name] = beta1 * m[var_name] + (1.0 - beta1) * g;
                    v[var_name] = beta2 * v[var_name] + (1.0 - beta2) * g * g;

                    const double m_hat = m[var_name] / one_minus_beta1_t;
                    const double v_hat = v[var_name] / one_minus_beta2_t;
                    theta[var_name] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
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

    class AdamW
    {
    private:
        double learning_rate;
        int max_iter;
        double beta1;
        double beta2;
        double epsilon;
        double weight_decay;

    public:
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

    // optimizer class that performs optimization for all optimizers

    class Optimizer
    {
    private:
        OptimParams optim_params;

    public:
        Optimizer(OptimParams optim_params) : optim_params(optim_params) {}

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

                // optimize the loss function

                std::unordered_map<std::string, double> theta = gd.optimize(loss, initial_theta);

                return theta;
            }

            else if (optim_params.get_optim_type() == OptimType::SGD)
            {
                SGD sgd(optim_params.get_learning_rate(), optim_params.get_max_iter());

                std::unordered_map<std::string, double> theta =
                    sgd.optimize(loss_function, xTr, yTr, initial_theta, optim_params.get_batch_size());
                return theta;
            }

            else if (optim_params.get_optim_type() == OptimType::ADAGRAD)
            {
                Adagrad adagrad(optim_params.get_learning_rate(), optim_params.get_max_iter(), optim_params.get_epsilon());

                std::unordered_map<std::string, double> theta =
                    adagrad.optimize(loss_function, xTr, yTr, initial_theta, optim_params.get_batch_size());
                return theta;
            }

            else if (optim_params.get_optim_type() == OptimType::RMSPROP)
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

            else if (optim_params.get_optim_type() == OptimType::ADAM)
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

            else if (optim_params.get_optim_type() == OptimType::ADAMW)
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

}
