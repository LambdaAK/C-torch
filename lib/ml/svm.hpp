#pragma once
#include "math/matrix.hpp"
#include "math/dataaugmentor.hpp"

namespace ctorch::distributed
{
    class ProcessGroup;
}

namespace ml {
    /**
     * @brief Binary linear support vector machine classifier.
     *
     * Uses a quadratic-program dual formulation with box constraints and
     * an equality constraint, solved numerically.
     * Training data layout:
     * - `xTr`: `(num_samples, num_features)`
     * - `yTr`: `(1, num_samples)` with labels typically in `{-1, 1}`
     */
    class SVM {
        private:
            Matrix xTr;                         ///< Cached training features.
            Matrix yTr;                         ///< Cached training labels.
            Matrix weights;                     ///< Learned weight row vector.
            double bias;                        ///< Learned bias term.
            DataAugmentationType augmentation_type; ///< Optional feature augmentation mode.

        public:
            /**
             * @brief Constructs an untrained SVM placeholder.
             */
            SVM();

            /**
             * @brief Trains an SVM on the provided data.
             * @param xTr Training features `(num_samples, num_features)`.
             * @param yTr Training labels `(1, num_samples)`.
             * @param learning_rate Step size hint for the QP solver.
             * @param max_iter Number of optimization iterations.
             * @param C Box-constraint constant for dual variables.
             * @param augmentation_type Feature augmentation mode.
             */
            SVM(Matrix xTr, Matrix yTr, double learning_rate, int max_iter, double C, DataAugmentationType augmentation_type);

            /**
             * @brief Trains a linear SVM with synchronous distributed gradient averaging on the primal hinge-loss objective.
             */
            static SVM train_distributed(
                Matrix xTr,
                Matrix yTr,
                double learning_rate,
                int max_iter,
                double C,
                DataAugmentationType augmentation_type,
                ctorch::distributed::ProcessGroup &group);

            /**
             * @brief Predicts the signed class for one sample.
             * @param x Input sample row matrix.
             * @return `1` if decision function is positive, otherwise `-1`.
             */
            int predict(const Matrix &x) const;

            /**
             * @brief Computes classification accuracy on test data.
             * @param xTe Test features `(num_samples, num_features)`.
             * @param yTe Test labels `(1, num_samples)`.
             * @return Fraction of correct predictions.
             */
            double score(const Matrix &xTe, const Matrix &yTe) const;
    };
}
