#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

#include "math/matrix.hpp"

namespace ml
{
    /**
     * @brief Binary perceptron classifier for linearly separable data.
     *
     * Input layout:
     * - `xTr`: `(num_samples, num_features)`
     * - `yTr`: `(1, num_samples)` with class labels in `{-1, 1}` or equivalent.
     */
    class Perceptron
    {
    private:
        Matrix weights; ///< Learned weight vector.
        double bias;    ///< Learned bias term.

    public:
        /**
         * @brief Trains perceptron weights over a fixed number of epochs.
         * @param xTr Training features `(num_samples, num_features)`.
         * @param yTr Training labels `(1, num_samples)`.
         * @param epochs Number of passes over training data.
         */
        Perceptron(Matrix xTr, Matrix yTr, int epochs = 300);

        /**
         * @brief Predicts class for a single sample.
         * @param x Input sample as a row matrix.
         * @return Predicted class label.
         */
        int predict(const Matrix &x) const;

        /**
         * @brief Computes classifier accuracy on a labeled test set.
         * @param xTe Test features `(num_samples, num_features)`.
         * @param yTe Test labels `(1, num_samples)`.
         * @return Fraction of correctly classified samples.
         */
        double score(const Matrix &xTe, const Matrix &yTe) const;
    };
}
