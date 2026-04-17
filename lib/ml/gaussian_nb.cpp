#include "gaussian_nb.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ml {

namespace {
constexpr double kVarFloor = 1e-6;
}

GaussianNB::GaussianNB(const Matrix& xTr, const Matrix& yTr) {
    if (yTr.numRows() != 1) {
        throw std::invalid_argument("GaussianNB: yTr must be a row vector (1, n).");
    }
    if (xTr.numRows() != yTr.numCols()) {
        throw std::invalid_argument("GaussianNB: xTr rows must match yTr columns.");
    }

    n_features = xTr.numCols();
    const size_t n = xTr.numRows();

    int max_label = 0;
    for (size_t i = 0; i < n; ++i) {
        int y = static_cast<int>(yTr(0, i));
        if (y < 0) {
            throw std::invalid_argument("GaussianNB: labels must be non-negative integers.");
        }
        if (y > max_label) {
            max_label = y;
        }
    }
    num_classes = max_label + 1;
    if (num_classes < 1) {
        throw std::invalid_argument("GaussianNB: no classes found.");
    }

    std::vector<size_t> counts(static_cast<size_t>(num_classes), 0);
    for (size_t i = 0; i < n; ++i) {
        counts[static_cast<size_t>(yTr(0, i))]++;
    }

    log_class_prior.resize(static_cast<size_t>(num_classes));
    for (int c = 0; c < num_classes; ++c) {
        if (counts[static_cast<size_t>(c)] == 0) {
            throw std::invalid_argument("GaussianNB: every class in [0, max_label] must appear at least once.");
        }
        log_class_prior[static_cast<size_t>(c)] =
            std::log(static_cast<double>(counts[static_cast<size_t>(c)]) / static_cast<double>(n));
    }

    mean_.assign(static_cast<size_t>(num_classes), std::vector<double>(n_features, 0.0));
    for (size_t i = 0; i < n; ++i) {
        int c = static_cast<int>(yTr(0, i));
        for (size_t j = 0; j < n_features; ++j) {
            mean_[static_cast<size_t>(c)][j] += xTr(i, j);
        }
    }
    for (int c = 0; c < num_classes; ++c) {
        const double inv = 1.0 / static_cast<double>(counts[static_cast<size_t>(c)]);
        for (size_t j = 0; j < n_features; ++j) {
            mean_[static_cast<size_t>(c)][j] *= inv;
        }
    }

    var_.assign(static_cast<size_t>(num_classes), std::vector<double>(n_features, 0.0));
    for (size_t i = 0; i < n; ++i) {
        int c = static_cast<int>(yTr(0, i));
        for (size_t j = 0; j < n_features; ++j) {
            const double d = xTr(i, j) - mean_[static_cast<size_t>(c)][j];
            var_[static_cast<size_t>(c)][j] += d * d;
        }
    }
    for (int c = 0; c < num_classes; ++c) {
        const double inv = 1.0 / static_cast<double>(counts[static_cast<size_t>(c)]);
        for (size_t j = 0; j < n_features; ++j) {
            double v = var_[static_cast<size_t>(c)][j] * inv;
            if (v < kVarFloor) {
                v = kVarFloor;
            }
            var_[static_cast<size_t>(c)][j] = v;
        }
    }
}

int GaussianNB::predict(const Matrix& x) const {
    if (x.numRows() != 1 || x.numCols() != n_features) {
        throw std::invalid_argument("GaussianNB::predict: x must be one row, numCols == num_features.");
    }

    int best_c = 0;
    double best_logp = -std::numeric_limits<double>::infinity();

    for (int c = 0; c < num_classes; ++c) {
        double logp = log_class_prior[static_cast<size_t>(c)];
        for (size_t j = 0; j < n_features; ++j) {
            const double mu = mean_[static_cast<size_t>(c)][j];
            const double v = var_[static_cast<size_t>(c)][j];
            const double diff = x(0, j) - mu;
            logp += -0.5 * std::log(v) - 0.5 * diff * diff / v;
        }
        if (logp > best_logp) {
            best_logp = logp;
            best_c = c;
        }
    }
    return best_c;
}

double GaussianNB::score(const Matrix& xTe, const Matrix& yTe) {
    if (yTe.numRows() != 1 || xTe.numRows() != yTe.numCols()) {
        throw std::invalid_argument("GaussianNB::score: xTe rows must match yTe columns count.");
    }

    size_t correct = 0;
    std::vector<Matrix> rows = xTe.rowsAsMatrices();
    for (size_t i = 0; i < xTe.numRows(); ++i) {
        if (predict(rows[i]) == static_cast<int>(yTe(0, i))) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(xTe.numRows());
}

} // namespace ml
