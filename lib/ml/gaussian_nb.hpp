#pragma once

#include <vector>
#include "math/matrix.hpp"

namespace ml {

/**
 * @brief Gaussian Naive Bayes classifier for continuous features.
 *
 * The model uses class-conditional independent normal distributions
 * (diagonal covariance) per feature. Training layout:
 * - `xTr`: `(num_samples, num_features)`
 * - `yTr`: `(1, num_samples)` with integer labels in `[0, num_classes - 1]`
 */
class GaussianNB {
public:
    /**
     * @brief Fits per-class priors, means, and variances from training data.
     * @param xTr Training features `(num_samples, num_features)`.
     * @param yTr Training label row vector `(1, num_samples)`.
     */
    GaussianNB(const Matrix& xTr, const Matrix& yTr);

    /**
     * @brief Predicts the most likely class for one sample.
     * @param x Input sample row matrix `(1, num_features)`.
     * @return Predicted class index.
     */
    int predict(const Matrix& x) const;

    /**
     * @brief Computes classification accuracy on a test set.
     * @param xTe Test features `(num_samples, num_features)`.
     * @param yTe Test labels `(1, num_samples)`.
     * @return Fraction of correctly predicted labels.
     */
    double score(const Matrix& xTe, const Matrix& yTe);

private:
    size_t n_features = 0;                   ///< Number of features per sample.
    int num_classes = 0;                     ///< Number of classes inferred from labels.
    std::vector<double> log_class_prior;     ///< Log prior probability per class.
    std::vector<std::vector<double>> mean_;  ///< Class-wise feature means.
    std::vector<std::vector<double>> var_;   ///< Class-wise feature variances (floored).
};

} // namespace ml
