#include "math/ast.hpp"
#include "math/matrix.hpp"
#include "math/optim.hpp"
#include "math/dataaugmentor.hpp"

using math::ASTNode;

namespace ml {
    /**
     * @brief Binary logistic regression classifier.
     *
     * Supports optional feature augmentation via `DataAugmentor` before training
     * and inference. Training data follows row-major sample layout:
     * - `xTr`: `(num_samples, num_features)`
     * - `yTr`: `(1, num_samples)` with labels in `{0, 1}`
     */
    class LogisticRegression {
        private:
            Matrix xTr;                         ///< Cached (possibly augmented) training features.
            Matrix yTr;                         ///< Cached training labels.
            std::vector<Matrix> xTr_rows;       ///< Row-wise cache of `xTr`.
            Matrix weights;                     ///< Learned weight row vector.
            double bias;                        ///< Learned bias term.
            math::OptimParams optim_params;     ///< Optimizer configuration used during fitting.

            /**
             * @brief Builds logistic loss for one labeled sample.
             * @param x Sample feature row matrix `(1, num_features)`.
             * @param y Binary label for the sample.
             * @return AST for binary cross-entropy sample loss.
             */
            std::shared_ptr<ASTNode> single_loss(const Matrix &x, int y) const;

            /**
             * @brief Builds mean logistic loss over all cached training samples.
             * @return AST representing average binary cross-entropy.
             */
            std::shared_ptr<ASTNode> loss() const;

            DataAugmentationType data_augmentation_type; ///< Feature augmentation strategy.
        
        public:
            /**
             * @brief Trains a logistic regression model with configurable optimizer.
             * @param xTr Training features `(num_samples, num_features)`.
             * @param yTr Training labels `(1, num_samples)` with values in `{0,1}`.
             * @param optim_params Optimizer settings.
             * @param data_augmentation_type Optional feature expansion strategy.
             */
            LogisticRegression(Matrix xTr, Matrix yTr, math::OptimParams optim_params, DataAugmentationType data_augmentation_type = DataAugmentationType::NO_OP);

            /**
             * @brief Predicts the binary class for one sample.
             * @param x Input sample row matrix `(1, num_features)` before augmentation.
             * @return `1` if predicted probability is at least `0.5`, otherwise `0`.
             */
            double predict(const Matrix &x) const;

            /**
             * @brief Computes classification accuracy over a test dataset.
             * @param xTe Test features `(num_samples, num_features)`.
             * @param yTe Test labels `(1, num_samples)`.
             * @return Fraction of correctly predicted labels.
             */
            double score(const Matrix &xTe, const Matrix &yTe) const;
    };
}
