#ifndef KNN_HPP
#define KNN_HPP

#include <vector>
#include "math/matrix.hpp"

namespace ml
{
    /**
     * @brief k-nearest neighbors classifier for row-vector samples.
     *
     * Training data layout:
     * - `xTr`: `(num_samples, num_features)`
     * - `yTr`: `(1, num_samples)`
     */
    class KNN
    {
    private:
        Matrix xTr; ///< Training feature matrix.
        Matrix yTr; ///< Training label row vector.
        size_t k;   ///< Number of nearest neighbors to vote over.

    public:
        /**
         * @brief Creates a KNN classifier with fixed training data.
         * @param k Number of neighbors used for voting.
         * @param xTr Training features `(num_samples, num_features)`.
         * @param yTr Training labels `(1, num_samples)`.
         */
        KNN(size_t k, Matrix xTr, Matrix yTr);

        /**
         * @brief Predicts the label for one sample.
         * @param x Input row matrix `(1, num_features)`.
         * @return Predicted class label.
         */
        int predict(const Matrix &x) const;

        /**
         * @brief Computes model accuracy on a test set.
         * @param xTe Test features `(num_samples, num_features)`.
         * @param yTe Test labels `(1, num_samples)`.
         * @return Fraction of correctly classified samples.
         */
        double score(const Matrix &xTe, const Matrix &yTe);

        /**
         * @brief Returns the currently configured neighborhood size.
         * @return Number of neighbors `k`.
         */
        size_t getK() const;

        /**
         * @brief Updates the neighborhood size used at inference.
         * @param new_k New number of neighbors.
         */
        void setK(size_t new_k);
    };

}

#endif // KNN_HPP
