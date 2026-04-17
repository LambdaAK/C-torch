#include "logisticregression.hpp"
#include "../math/matrix.hpp"
#include "../math/ast.hpp"
#include "../math/optim.hpp"
#include "../math/lossfunction.hpp"
#include "../math/dataaugmentor.hpp"

using math::ASTNode;
using math::GD;
using math::Num;
using math::Optimizer;
using math::OptimParams;
using math::OptimType;
using math::SGD;
using math::Var;

namespace ml
{

    /**
     * Compute the loss function for one sample in logistic regression
     */
    std::shared_ptr<ASTNode> LogisticRegression::single_loss(const Matrix &x, int y) const
    {
        /*
            The weights are w1, ..., wn and the bias is b
            The prediction is sigmoid(w1 * x1 + ... + wn * xn + b)
            The single sample loss function is -y * log(prediction) - (1 - y) * log(1 - prediction)
        */

        std::shared_ptr<ASTNode> w_transpose_x = Num(0);

        for (size_t i = 0; i < x.numCols(); i++)
        {
            std::shared_ptr<ASTNode> w_i = Var("w" + std::to_string(i));
            std::shared_ptr<ASTNode> x_i = Num(x.at(0, i));
            w_transpose_x = w_transpose_x + w_i * x_i;
        }

        std::shared_ptr<ASTNode> b = Var("b");

        std::shared_ptr<ASTNode> w_transpose_x_plus_b = w_transpose_x + b;

        std::shared_ptr<ASTNode> y_hat = Sigmoid(w_transpose_x_plus_b); // predicted label

        // loss is -y * log(y_hat) - (1 - y) * log(1 - y_hat)

        std::shared_ptr<ASTNode> loss = -Num(y) * Log(y_hat) - (Num(1) - Num(y)) * Log(Num(1) - y_hat);

        return loss;
    }

    std::shared_ptr<ASTNode> LogisticRegression::loss() const
    {
        std::shared_ptr<ASTNode> loss = Num(0);
        for (size_t i = 0; i < xTr.numRows(); i++)
        {
            Matrix x = xTr_rows[i];
            int y = yTr.at(0, i);
            loss = loss + single_loss(x, y);
        }

        // divide by the number of samples
        std::shared_ptr<ASTNode> num_samples = Num(xTr.numRows());
        loss = loss / num_samples;
        // negative sign
        return loss;
    }

    // loss function class for logistic regression

    class LogisticLossFunction : public LossFunction
    {
        std::shared_ptr<ASTNode> sample_loss(const Matrix &x, int y) const override
        {
            /*
                The weights are w1, ..., wn and the bias is b
                The prediction is sigmoid(w1 * x1 + ... + wn * xn + b)
                The single sample loss function is -y * log(prediction) - (1 - y) * log(1 - prediction)
            */

            std::shared_ptr<ASTNode> w_transpose_x = Num(0);

            for (size_t i = 0; i < x.numCols(); i++)
            {
                std::shared_ptr<ASTNode> w_i = Var("w" + std::to_string(i));
                std::shared_ptr<ASTNode> x_i = Num(x.at(0, i));
                w_transpose_x = w_transpose_x + w_i * x_i;
            }

            std::shared_ptr<ASTNode> b = Var("b");

            std::shared_ptr<ASTNode> w_transpose_x_plus_b = w_transpose_x + b;

            std::shared_ptr<ASTNode> y_hat = Sigmoid(w_transpose_x_plus_b); // predicted label

            // loss is -y * log(y_hat) - (1 - y) * log(1 - y_hat)

            std::shared_ptr<ASTNode> loss = -Num(y) * Log(y_hat) - (Num(1) - Num(y)) * Log(Num(1) - y_hat);

            return loss;
        }

        std::shared_ptr<ASTNode> regularizer() const override
        {
            // no regularization for logistic regression
            return Num(0);
        }
    };

    LogisticRegression::LogisticRegression(
        Matrix xTr,
        Matrix yTr,
        OptimParams optim_params,
        DataAugmentationType data_augmentation_type) : yTr(yTr),
                                                       optim_params(optim_params),
                                                       data_augmentation_type(data_augmentation_type)
    {
        // check the dimensions
        if (xTr.numRows() != yTr.numCols())
        {
            throw std::invalid_argument("Number of training samples and labels must be equal.");
        }

        if (yTr.numRows() != 1)
        {
            throw std::invalid_argument("Labels must be a row vector.");
        }

        // cache the rows of xTr as matrices

        // augment the data

        xTr = DataAugmentor::augment_data(xTr, data_augmentation_type);
        this->xTr = xTr;

        // xTr.print();

        xTr_rows = xTr.rowsAsMatrices();

        // initialize the weights and bias

        weights = Matrix(1, xTr.numCols());

        // compute the loss function for logistic regression

        // optimize the loss function with respect to the weights and bias

        std::unordered_map<std::string, double> values;

        for (size_t i = 0; i < this->xTr.numCols(); i++)
        {
            std::string w_i = "w" + std::to_string(i);
            values[w_i] = 5;
        }

        values["b"] = 1;

        // optimize the parameters using the specified optimizer

        std::unordered_map<std::string, double> result;

        // optimize the loss function

        Optimizer optim(optim_params);

        std::shared_ptr<LossFunction> loss_function = std::make_shared<LogisticLossFunction>();

        result = optim.optimize(loss_function, this->xTr, yTr, values);

        // first, initialize weights to be a matrix of proper dimensions

        weights = Matrix(1, xTr.numCols());

        // assign the values accordingly from result

        for (size_t i = 0; i < this->xTr.numCols(); i++)
        {
            std::string w_i = "w" + std::to_string(i);
            weights(0, i) = result[w_i];
        }

        bias = result["b"];

        /*
            Now, the weights and bias are set accordingly
            The model is ready to be used!
        */

        // DEBUG: print the weights and biases

        // std::cout << "Weights: " << std::endl;
        // weights.print();
        // std::cout << "Bias: " << bias << std::endl;
    }

    double sigmoid(double x)
    {
        return 1 / (1 + std::exp(-x));
    }

    double LogisticRegression::predict(const Matrix &x) const
    {
        /*
            The prediction is 1 when sigmoid(w1 * x1 + ... + wn * xn + b) >= 0.5 and 0 otherwise
        */

        double result = 0;

        Matrix x_augmented = DataAugmentor::augment_data(x, data_augmentation_type);

        if (x_augmented.numCols() != weights.numCols())
        {
            throw std::invalid_argument("Feature dimension mismatch between input and trained weights.");
        }

        for (size_t i = 0; i < x_augmented.numCols(); i++)
        {
            result += weights(0, i) * x_augmented.at(0, i);
        }

        result += bias;

        result = sigmoid(result);

        return result >= 0.5 ? 1 : 0;
    }

    double LogisticRegression::score(const Matrix &xTe, const Matrix &yTe) const
    {
        if (yTe.numRows() != 1)
        {
            throw std::invalid_argument("Labels must be a row vector (1 × N).");
        }

        if (xTe.numRows() != yTe.numCols())
        {
            throw std::invalid_argument("Mismatch between number of test samples and labels.");
        }

        int correct = 0;
        for (size_t i = 0; i < xTe.numRows(); ++i)
        {
            Matrix x_i(1, xTe.numCols());
            for (size_t j = 0; j < xTe.numCols(); ++j)
            {
                x_i(0, j) = xTe(i, j);
            }

            int prediction = static_cast<int>(predict(x_i));
            int actual = static_cast<int>(yTe(0, i));

            if (prediction == actual)
            {
                correct++;
            }
        }

        return static_cast<double>(correct) / xTe.numRows();
    }

}
