#include "perceptron.hpp"

namespace ml
{

    Perceptron::Perceptron(Matrix xTr, Matrix yTr, int epochs)
    {
        /*
            Labels are +- 1
        */
        // check dimensions
        if (xTr.numRows() != yTr.numCols())
        {
            throw std::invalid_argument("Mismatch between number of training samples and labels.");
        };
        if (yTr.numRows() != 1)
        {
            throw std::invalid_argument("Labels must be a row vector.");
        };

        // save the rows of xTr as matrices

        std::vector<Matrix> xTr_rows = xTr.rowsAsMatrices();

        // initialize weights and bias.
        weights = Matrix(1, xTr.numCols());
        bias = 0.0;

        // train, using the perceptron algorithm
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            bool updated = false;
            for (size_t row = 0; row < xTr.numRows(); ++row)
            {
                Matrix x = xTr_rows[row]; // test vector

                int label = yTr.at(0, row); // label

                double prediction = predict(x);

                if (prediction != label)
                {
                    // update the weights and bias
                    updated = true;
                    for (size_t i = 0; i < x.numCols(); ++i)
                    {
                        weights(0, i) += label * x.at(0, i);
                    }
                    bias += label;
                }
            }
            if (!updated)
            {
                // we have found a linear decision boundary
                break;
            }
        }
    }

    /**
    Helper: returns the sign of a given value.
    **/
    int sign(double x)
    {
        return (x > 0) ? 1 : -1;
    }

    int Perceptron::predict(const Matrix &x) const
    {
        // use weights and bias
        // (w x ^ T) + b
        return sign((weights.inner_product(x) + bias));
    }

    double Perceptron::score(const Matrix &xTe, const Matrix &yTe) const
    {
        // check dimensions
        if (xTe.numRows() != yTe.numCols())
        {
            throw std::invalid_argument("Mismatch between number of test samples and labels.");
        }

        if (yTe.numRows() != 1)
        {
            throw std::invalid_argument("`yTe must be a row vector.");
        };

        std::vector<Matrix> xTe_row = xTe.rowsAsMatrices();

        int correct = 0;
        int total = xTe.numRows();

        for (size_t i = 0; i < total; ++i)
        {
            int prediction = predict(xTe_row[i]);
            int actual = (yTe.at(0, i) > 0) ? 1 : -1;
            if (prediction == actual)
            {
                correct++;
            }
        }

        return static_cast<double>(correct) / static_cast<double>(total);
    }

}