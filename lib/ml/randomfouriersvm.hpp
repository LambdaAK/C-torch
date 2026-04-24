#pragma once
#include "math/matrix.hpp"
#include "svm.hpp"

namespace ctorch::distributed
{
    class ProcessGroup;
}

namespace ml
{

    /**
     * @brief Kernel approximation SVM using Random Fourier Features (RFF).
     *
     * Maps input features into a randomized finite-dimensional feature space,
     * then trains a linear `SVM` in that transformed space.
     */
    class RandomFourierSVM
    {
    private:
        RandomFourierSVM() = default;

        int D;       ///< Number of random Fourier features.
        double gamma;///< RBF kernel bandwidth factor used in RFF mapping.
        Matrix W;    ///< Random projection matrix.
        Matrix b;    ///< Random phase offsets.
        SVM svm;     ///< Linear SVM trained in transformed feature space.

        /**
         * @brief Applies the stored random Fourier transform to input data.
         * @param x Input matrix.
         * @return Transformed feature matrix used by the inner SVM.
         */
        Matrix transform_data(const Matrix &x) const;

    public:
        /**
         * @brief Constructs and trains the RFF + linear SVM pipeline.
         * @param xTr Training features.
         * @param yTr Training labels.
         * @param D Number of random Fourier features.
         * @param gamma RBF bandwidth factor for feature mapping.
         * @param learning_rate Learning rate for the internal SVM optimizer.
         * @param max_iter Maximum iterations for SVM optimization.
         * @param C SVM hinge-loss penalty coefficient.
         */
        RandomFourierSVM(Matrix xTr, Matrix yTr, int D, double gamma, double learning_rate, int max_iter, double C);

        /**
         * @brief Trains the RFF pipeline with synchronous distributed training for the inner SVM.
         */
        static RandomFourierSVM train_distributed(
            Matrix xTr,
            Matrix yTr,
            int D,
            double gamma,
            double learning_rate,
            int max_iter,
            double C,
            ctorch::distributed::ProcessGroup &group);

        /**
         * @brief Predicts class label for one sample.
         * @param x Input sample row matrix.
         * @return Predicted signed class label from the inner SVM.
         */
        int predict(const Matrix &x) const;

        /**
         * @brief Computes classification accuracy on test data.
         * @param xTe Test features.
         * @param yTe Test labels.
         * @return Fraction of correctly predicted samples.
         */
        double score(const Matrix &xTe, const Matrix &yTe) const;
    };

}
