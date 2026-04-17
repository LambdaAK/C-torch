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

} // namespace math

