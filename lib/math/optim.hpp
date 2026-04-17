#pragma once

#include "ast.hpp"
#include <unordered_map>
#include "lossfunction.hpp"

namespace math
{

    enum class OptimType
    {
        GD,
        SGD
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

    public:
        OptimParams(OptimType optim_type = OptimType::GD, double learning_rate = 0.001, int max_iter = 1000, Matrix xTr = {{}}, Matrix yTr = {}, int batch_size = 1)
            : optim_type(optim_type), learning_rate(learning_rate), max_iter(max_iter), xTr(xTr), yTr(yTr), batch_size(batch_size) {}

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
    };

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

        std::unordered_map<std::string, double> optimize(std::shared_ptr<LossFunction> loss_function, Matrix xTr, Matrix yTr, std::unordered_map<std::string, double> initial_theta, int batch_size)
        {
            std::unordered_map<std::string, double> theta = initial_theta;

            Differentiator diff;

            for (int i = 0; i < max_iter; i++)
            {
                if (i % 20 == 0)
                {
                    std::cout << "Epoch: " << i << std::endl;
                }
                // sample a batch of data
                std::vector<int> indices;
                for (int j = 0; j < batch_size; j++)
                {
                    int index = rand() % xTr.numRows();
                    indices.push_back(index);
                }
                // compute the loss function for this epoch
                std::shared_ptr<ASTNode> loss = Num(0);

                std::vector<Matrix> xTr_rows = xTr.rowsAsMatrices();

                for (int j = 0; j < batch_size; j++)
                {
                    int index = indices[j];
                    Matrix x = xTr_rows[index];
                    int y = yTr.at(0, index);
                    std::shared_ptr<ASTNode> sample_loss = loss_function->sample_loss(x, y);
                    loss = loss + sample_loss;
                }

                // divide by the batch size

                loss = loss / Num(batch_size);

                // compute the regularizer

                std::shared_ptr<ASTNode> regularizer = loss_function->regularizer();

                // add the regularizer to the loss

                loss = loss + regularizer;

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

    // optimizer class that performs optimization for all optimizers

    class Optimizer
    {
    private:
        OptimParams optim_params;

    public:
        Optimizer(OptimParams optim_params) : optim_params(optim_params) {}

        std::unordered_map<std::string, double> optimize(std::shared_ptr<LossFunction> loss_function, Matrix xTr, Matrix yTr, std::unordered_map<std::string, double> initial_theta)
        {
            if (optim_params.get_optim_type() == OptimType::GD)
            {
                GD gd(optim_params.get_learning_rate(), optim_params.get_max_iter());

                // construct the full loss function

                std::shared_ptr<ASTNode> loss = Num(0);

                std::vector<Matrix> xTr_rows = xTr.rowsAsMatrices();

                for (size_t i = 0; i < xTr.numRows(); i++)
                {
                    Matrix x = xTr_rows[i];
                    int y = yTr.at(0, i);
                    std::shared_ptr<ASTNode> sample_loss = loss_function->sample_loss(x, y);
                    loss = loss + sample_loss;
                }

                // divide by the data set cardinality

                loss = loss / Num(xTr.numRows());

                //  finally, compute and add the regularizer

                std::shared_ptr<ASTNode> regularizer = loss_function->regularizer();

                loss = loss + regularizer;

                // optimize the loss function

                std::unordered_map<std::string, double> theta = gd.optimize(loss, initial_theta);

                return theta;
            }

            else if (optim_params.get_optim_type() == OptimType::SGD)
            {
                SGD sgd(optim_params.get_learning_rate(), optim_params.get_max_iter());

                std::unordered_map<std::string, double> theta = sgd.optimize(loss_function, optim_params.get_xTr(), optim_params.get_yTr(), initial_theta, optim_params.get_batch_size());
                return theta;
            }

            throw std::runtime_error("Unknown optimizer type");
        }
    };

}
