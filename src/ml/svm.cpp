#include "svm.hpp"
#include "../math/optim.hpp"
#include "../math/dataaugmentor.hpp"

using math::ASTNode;
using math::GD;
using math::Num;
using math::Var;

namespace ml
{

    SVM::SVM()
    {
        // default constructor
    }

    std::shared_ptr<ASTNode> SVM::loss_function() const
    {
        std::shared_ptr<ASTNode> w_t_w = Num(0);

        // add w ^ T w

        for (size_t i = 0; i < weights.numCols(); i++)
        {
            std::shared_ptr<ASTNode> w_i = Var("w" + std::to_string(i));
            w_t_w = w_t_w + w_i * w_i;
        }

        // add the sum of the hinge losses weighted by C, the cost constant

        std::shared_ptr<ASTNode> hinge_losses = Num(0);

        for (size_t i = 0; i < xTr.numRows(); i++)
        {
            Matrix x = xTr_rows[i];
            int y = yTr.at(0, i);

            // compute w ^ T x + b

            std::shared_ptr<ASTNode> w_transpose_x = Num(0);

            for (size_t j = 0; j < x.numCols(); j++)
            {
                std::shared_ptr<ASTNode> w_j = Var("w" + std::to_string(j));
                std::shared_ptr<ASTNode> x_j = Num(x.at(0, j));
                w_transpose_x = w_transpose_x + w_j * x_j;
            }

            // add b

            std::shared_ptr<ASTNode> b = Var("b");

            std::shared_ptr<ASTNode> w_transpose_x_plus_b = w_transpose_x + b;

            // compute the hinge loss

            std::shared_ptr<ASTNode> hinge_loss = Max(Num(0), Num(1) - Num(y) * w_transpose_x_plus_b);

            hinge_losses = hinge_losses + hinge_loss;
        }

        // construct the total loss function

        std::shared_ptr<ASTNode> C = Num(this->C);

        std::shared_ptr<ASTNode> loss = w_t_w + C * hinge_losses;

        return loss;
    }

    SVM::SVM(Matrix xTr, Matrix yTr, double learning_rate, int max_iter, double C, DataAugmentationType augmentation_type) : xTr(xTr), yTr(yTr), learning_rate(learning_rate), max_iter(max_iter), C(C), augmentation_type(augmentation_type)
    {
        // check the dimensions of the matrices
        if (xTr.numRows() != yTr.numCols())
        {
            throw std::invalid_argument("Number of training samples and labels must be equal.");
        }

        if (yTr.numRows() != 1)
        {
            throw std::invalid_argument("Labels must be a row vector.");
        }

        // augment the data

        xTr = DataAugmentor::augment_data(xTr, augmentation_type);

        // xTr.print();

        // cache the rows of xTr as matrices
        xTr_rows = xTr.rowsAsMatrices();

        // initialize the weights and bias

        weights = Matrix(1, xTr.numCols());
        bias = 0;

        // compute the loss function

        std::shared_ptr<ASTNode> loss = loss_function();

        // optimize the loss function with respect to the weights and bias

        std::unordered_map<std::string, double> values;

        for (size_t i = 0; i < xTr.numCols(); i++)
        {
            std::string w_i = "w" + std::to_string(i);
            values[w_i] = 1;
        }

        values["b"] = 1;

        GD gd(learning_rate, max_iter);

        values = gd.optimize(loss, values);

        // update the weights and bias

        for (size_t i = 0; i < xTr.numCols(); i++)
        {
            std::string w_i = "w" + std::to_string(i);
            weights(0, i) = values[w_i];
        }

        bias = values["b"];
    }

    int SVM::predict(const Matrix &x) const
    {

        // augment x

        const Matrix x_augmented = DataAugmentor::augment_data(x, augmentation_type);

        // compute sign(w ^ T x + b)
        double a = 0;

        a += weights.inner_product(x_augmented);

        a += bias;

        return (a > 0) ? 1 : -1;
    }

    double SVM::score(const Matrix &xTe, const Matrix &yTe) const
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

            int prediction = predict(x_i);
            int actual = static_cast<int>(yTe(0, i));
            if (prediction == actual)
            {
                correct++;
            }
        }

        return static_cast<double>(correct) / xTe.numRows();
    }

}