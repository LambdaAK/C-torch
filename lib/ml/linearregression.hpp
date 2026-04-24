#include "math/matrix.hpp"
#include "math/ast.hpp"

using math::ASTNode;

namespace ctorch::distributed
{
    class ProcessGroup;
}

namespace ml
{
    /**
     * @brief Linear regression model trained with gradient descent on OLS loss.
     *
     * Training data is expected in row-major sample layout:
     * - `xTr`: shape `(num_samples, num_features)`
     * - `yTr`: shape `(1, num_samples)`
     */
    class LinearRegression
    {
    private:
        LinearRegression() = default;

        Matrix xTr;                    ///< Cached training features.
        std::vector<Matrix> xTr_rows;  ///< Row-wise view of `xTr` for loss construction.
        Matrix yTr;                    ///< Cached training labels.
        Matrix weights;                ///< Learned weight row vector of shape `(1, num_features)`.
        double bias;                   ///< Learned bias term.
        double learning_rate;          ///< Gradient descent learning rate.
        int max_iter;                  ///< Maximum optimization iterations.

        /**
         * @brief Builds the squared loss expression for one sample.
         * @param x Single sample row matrix of shape `(1, num_features)`.
         * @param y Scalar label for the sample.
         * @return AST representing `(y - (w^T x + b))^2`.
         */
        std::shared_ptr<ASTNode> single_squared_loss(const Matrix &x, double y) const;

        /**
         * @brief Builds average ordinary least squares (OLS) loss over training data.
         * @return AST representing mean sample squared loss.
         */
        std::shared_ptr<ASTNode> OLS_loss() const;

    public:
        /**
         * @brief Trains a linear regression model on the provided dataset.
         * @param xTr Training features `(num_samples, num_features)`.
         * @param yTr Training labels `(1, num_samples)`.
         * @param learning_rate Step size used by gradient descent.
         * @param max_iter Number of gradient descent iterations.
         */
        LinearRegression(Matrix xTr, Matrix yTr, double learning_rate, int max_iter);

        /**
         * @brief Trains a linear regression model with synchronous distributed gradient averaging.
         */
        static LinearRegression train_distributed(
            Matrix xTr,
            Matrix yTr,
            double learning_rate,
            int max_iter,
            ctorch::distributed::ProcessGroup &group);

        /**
         * @brief Predicts a continuous target for one sample.
         * @param x Input sample row matrix `(1, num_features)`.
         * @return Predicted scalar value `w^T x + b`.
         */
        double predict(const Matrix &x) const;

        /**
         * @brief Computes binary accuracy using a threshold over regression outputs.
         * @param xTe Test features `(num_samples, num_features)`.
         * @param yTe Test labels `(1, num_samples)`.
         * @param threshold Values lower than threshold are mapped to class `1`, else `0`.
         * @return Fraction of correctly classified test samples.
         */
        double score(const Matrix &xTe, const Matrix &yTe, double threshold = 0.5) const;
    };
}
