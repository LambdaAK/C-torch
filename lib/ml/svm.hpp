#pragma once
#include "math/matrix.hpp"
#include "math/ast.hpp"
#include "math/dataaugmentor.hpp"

using math::ASTNode;

namespace ml {
    /**
     * @brief Binary linear support vector machine classifier.
     *
     * Uses hinge-loss objective optimized through symbolic differentiation.
     * Training data layout:
     * - `xTr`: `(num_samples, num_features)`
     * - `yTr`: `(1, num_samples)` with labels typically in `{-1, 1}`
     */
    class SVM {
        private:
            Matrix xTr;                         ///< Cached training features.
            Matrix yTr;                         ///< Cached training labels.
            std::vector<Matrix> xTr_rows;       ///< Row-wise cache of training features.
            Matrix weights;                     ///< Learned weight row vector.
            double bias;                        ///< Learned bias term.
            double learning_rate;               ///< Optimization step size.
            int max_iter;                       ///< Number of optimization iterations.
            double C;                           ///< Hinge-loss penalty coefficient.
            DataAugmentationType augmentation_type; ///< Optional feature augmentation mode.

            /**
             * @brief Builds hinge-loss expression for one sample.
             * @param x Sample feature row matrix.
             * @param y Signed class label.
             * @return AST for a single-sample hinge-loss term.
             */
            std::shared_ptr<ASTNode> single_sample_loss(const Matrix &x, int y) const;

            /**
             * @brief Builds full SVM objective (margin regularizer + hinge loss).
             * @return AST representing the model training objective.
             */
            std::shared_ptr<ASTNode> loss_function() const;

        public:
            /**
             * @brief Constructs an untrained SVM placeholder.
             */
            SVM();

            /**
             * @brief Trains an SVM on the provided data.
             * @param xTr Training features `(num_samples, num_features)`.
             * @param yTr Training labels `(1, num_samples)`.
             * @param learning_rate Gradient descent step size.
             * @param max_iter Number of optimization iterations.
             * @param C Hinge-loss regularization constant.
             * @param augmentation_type Feature augmentation mode.
             */
            SVM(Matrix xTr, Matrix yTr, double learning_rate, int max_iter, double C, DataAugmentationType augmentation_type);

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
