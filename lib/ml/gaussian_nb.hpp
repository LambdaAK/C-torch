#pragma once

#include <vector>
#include "math/matrix.hpp"

namespace ml {

/**
 * Gaussian Naive Bayes for continuous features (diagonal covariance per class).
 * Training layout matches KNN: rows of xTr are samples, yTr is shape (1, n) with
 * integer labels in [0, num_classes - 1].
 */
class GaussianNB {
public:
    GaussianNB(const Matrix& xTr, const Matrix& yTr);

    int predict(const Matrix& x) const;

    double score(const Matrix& xTe, const Matrix& yTe);

private:
    size_t n_features = 0;
    int num_classes = 0;
    std::vector<double> log_class_prior;
    std::vector<std::vector<double>> mean_;
    std::vector<std::vector<double>> var_;
};

} // namespace ml
