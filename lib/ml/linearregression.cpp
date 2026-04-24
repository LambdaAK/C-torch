#include "linearregression.hpp"
#include "distributed/distributed_optimizer.hpp"
#include "math/optim.hpp"
#include "math/ast.hpp"
#include "math/matrix.hpp"
#include "math/optim.hpp"
#include <stdexcept>

using math::ASTNode;
using math::GD;
using math::Num;
using math::Var;

namespace
{
class LinearRegressionLossFunction final : public LossFunction
{
public:
    std::shared_ptr<ASTNode> sample_loss(const Matrix &x, double y) const override
    {
        std::shared_ptr<ASTNode> w_transpose_x = Num(0);
        for (size_t i = 0; i < x.numCols(); ++i)
        {
            std::shared_ptr<ASTNode> w_i = Var("w" + std::to_string(i));
            std::shared_ptr<ASTNode> x_i = Num(x.at(0, i));
            w_transpose_x = w_transpose_x + w_i * x_i;
        }

        std::shared_ptr<ASTNode> b = Var("b");
        std::shared_ptr<ASTNode> y_hat = w_transpose_x + b;
        return (Num(y) - y_hat) ^ Num(2);
    }

    std::shared_ptr<ASTNode> regularizer() const override
    {
        return Num(0);
    }
};
} // namespace

namespace ml
{
    LinearRegression LinearRegression::train_distributed(
        Matrix xTr,
        Matrix yTr,
        double learning_rate,
        int max_iter,
        ctorch::distributed::ProcessGroup &group)
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
            throw std::invalid_argument("Number of training samples and labels must be equal.");
        }
        if (learning_rate <= 0.0)
        {
            throw std::invalid_argument("learning_rate must be positive.");
        }
        if (max_iter <= 0)
        {
            throw std::invalid_argument("max_iter must be positive.");
        }

        LinearRegression model;
        model.xTr = xTr;
        model.yTr = yTr;
        model.learning_rate = learning_rate;
        model.max_iter = max_iter;
        model.xTr_rows = xTr.rowsAsMatrices();
        model.weights = Matrix(1, xTr.numCols());
        model.bias = 0.0;

        std::unordered_map<std::string, double> values;
        for (size_t i = 0; i < xTr.numCols(); i++)
        {
            std::string w_i = "w" + std::to_string(i);
            values[w_i] = 1;
        }
        values["b"] = 1;

        std::shared_ptr<LossFunction> loss_function = std::make_shared<LinearRegressionLossFunction>();
        math::OptimParams optim_params(math::OptimType::GD, learning_rate, max_iter);
        ctorch::distributed::DistributedOptimizer optimizer(group, optim_params);
        std::unordered_map<std::string, double> theta = optimizer.optimize(loss_function, model.xTr, model.yTr, values);

        for (size_t i = 0; i < xTr.numCols(); i++)
        {
            std::string w_i = "w" + std::to_string(i);
            model.weights(0, i) = theta[w_i];
        }

        model.bias = theta["b"];
        return model;
    }

    std::shared_ptr<ASTNode> LinearRegression::single_squared_loss(const Matrix &x, double y) const
    {
        /*
            The weights are w1, ...., wn and the bias is b

            The prediction is w1 * x1 + ... + wn * xn + b
        */

        // compute w1 * x1 + ... + wn * xn + b

        std::shared_ptr<ASTNode> w_transpose_x = Num(0);

        for (size_t i = 0; i < x.numCols(); i++)
        {
            std::shared_ptr<ASTNode> w_i = Var("w" + std::to_string(i));
            std::shared_ptr<ASTNode> x_i = Num(x.at(0, i));
            w_transpose_x = w_transpose_x + w_i * x_i;
        }

        std::shared_ptr<ASTNode> b = Var("b");

        // the prediction is w_transpose_x + b

        std::shared_ptr<ASTNode> y_hat = w_transpose_x + b;

        // compute the squared loss between the label y and prediction y_hat

        std::shared_ptr<ASTNode> loss = (Num(y) - y_hat) ^ Num(2);

        return loss;
    }

    std::shared_ptr<ASTNode> LinearRegression::OLS_loss() const
    {
        std::shared_ptr<ASTNode> loss = Num(0);

        for (size_t i = 0; i < xTr.numRows(); i++)
        {
            Matrix x = xTr_rows[i];
            double y = yTr.at(0, i);
            loss = loss + single_squared_loss(x, y);
        }

        // divide by the data set size

        loss = loss / Num(xTr.numRows());

        return loss;
    }

    LinearRegression::LinearRegression(Matrix xTr, Matrix yTr, double learning_rate, int max_iter)
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
            throw std::invalid_argument("Number of training samples and labels must be equal.");
        }
        if (learning_rate <= 0.0)
        {
            throw std::invalid_argument("learning_rate must be positive.");
        }
        if (max_iter <= 0)
        {
            throw std::invalid_argument("max_iter must be positive.");
        }

        // cache everything
        this->xTr = xTr;
        this->yTr = yTr;
        this->learning_rate = learning_rate;
        this->max_iter = max_iter;

        // cache the rows of xTr as matrices

        xTr_rows = xTr.rowsAsMatrices();

        // initialize the weights and bias

        weights = Matrix(1, xTr.numCols());
        bias = 0;

        // compute the loss function

        std::shared_ptr<ASTNode> loss_function = OLS_loss();

        // optimize the loss function with respect to the weights and bias

        // first, initialize the parameter values

        std::unordered_map<std::string, double> values;

        for (size_t i = 0; i < xTr.numCols(); i++)
        {
            std::string w_i = "w" + std::to_string(i);
            values[w_i] = 1;
        }

        values["b"] = 1;

        // optimize

        GD gd(learning_rate, max_iter);

        std::unordered_map<std::string, double> theta = gd.optimize(loss_function, values);

        // update the weights and biases

        for (size_t i = 0; i < xTr.numCols(); i++)
        {
            std::string w_i = "w" + std::to_string(i);
            weights(0, i) = theta[w_i];
        }

        bias = theta["b"];

        /*
            The parameters have been optimized
            The model is ready to be used
        */

        // DEBUG: print the weights and biases

        std::cout << "Weights: " << std::endl;
        weights.print();

        std::cout << "Bias: " << bias << std::endl;
    }

    double LinearRegression::predict(const Matrix &x) const
    {
        if (x.numRows() != 1)
        {
            throw std::invalid_argument("x must be a row vector (1 x D).");
        }
        if (x.numCols() != weights.numCols())
        {
            throw std::invalid_argument("Feature dimension mismatch between input and trained weights.");
        }

        // compute w ^ T x + b

        double inner_prod = 0;

        for (size_t i = 0; i < weights.numCols(); i++)
        {
            inner_prod += weights(0, i) * x(0, i);
        }

        double y_hat = inner_prod + bias;

        return y_hat;
    }

    double LinearRegression::score(const Matrix &xTe, const Matrix &yTe, double threshold) const
    {
        // Check that yTe is a row vector
        if (yTe.numRows() != 1)
        {
            throw std::invalid_argument("yTe must be a row vector");
        }
        if (xTe.numRows() != yTe.numCols())
        {
            throw std::invalid_argument("Number of test samples and labels must be equal.");
        }
        if (xTe.numCols() != weights.numCols())
        {
            throw std::invalid_argument("Test feature dimension must match training feature dimension.");
        }
        if (xTe.numRows() == 0)
        {
            throw std::invalid_argument("xTe must contain at least one sample.");
        }

        double correct_count = 0.0;
        size_t total_samples = xTe.numRows();

        for (size_t i = 0; i < xTe.numRows(); ++i)
        {
            Matrix x_sample(1, xTe.numCols());
            for (size_t j = 0; j < xTe.numCols(); ++j)
            {
                x_sample(0, j) = xTe(i, j);
            }

            double prediction = predict(x_sample);

            int binary_prediction = (prediction < threshold) ? 1 : 0;

            if (binary_prediction == static_cast<int>(yTe(0, i)))
            {
                correct_count += 1.0;
            }
        }

        return correct_count / total_samples;
    }

}
