#include "dataaugmentor.hpp"
#include "parallel.hpp"
#include <random>
#include <cmath>
#include <limits>
#include <stdexcept>

// Add this if M_PI is not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Matrix DataAugmentor::no_op(const Matrix &x) {
    return x;
}

Matrix DataAugmentor::poly_2(const Matrix &x) {
    Matrix result(x.numRows(), x.numCols() * 2);
    ctorch::parallel::parallel_for_items(x.numRows(), x.numCols(), [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            for (size_t j = 0; j < x.numCols(); ++j) {
                const double value = x(i, j);
                result(i, j) = value;
                result(i, j + x.numCols()) = value * value;
            }
        }
    });
    return result;
}

Matrix DataAugmentor::poly_3(const Matrix &x) {
    Matrix result(x.numRows(), x.numCols() * 3);
    ctorch::parallel::parallel_for_items(x.numRows(), x.numCols(), [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            for (size_t j = 0; j < x.numCols(); ++j) {
                const double value = x(i, j);
                result(i, j) = value;
                result(i, j + x.numCols()) = value * value;
                result(i, j + 2 * x.numCols()) = value * value * value;
            }
        }
    });
    return result;
}

Matrix DataAugmentor::poly_4(const Matrix &x) {
    Matrix result(x.numRows(), x.numCols() * 4);
    ctorch::parallel::parallel_for_items(x.numRows(), x.numCols(), [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            for (size_t j = 0; j < x.numCols(); ++j) {
                const double value = x(i, j);
                const double squared = value * value;
                result(i, j) = value;
                result(i, j + x.numCols()) = squared;
                result(i, j + 2 * x.numCols()) = squared * value;
                result(i, j + 3 * x.numCols()) = squared * squared;
            }
        }
    });
    return result;
}

Matrix DataAugmentor::poly_5(const Matrix &x) {
    Matrix result(x.numRows(), x.numCols() * 5);
    ctorch::parallel::parallel_for_items(x.numRows(), x.numCols(), [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            for (size_t j = 0; j < x.numCols(); ++j) {
                const double value = x(i, j);
                const double squared = value * value;
                const double cubed = squared * value;
                const double fourth = squared * squared;
                result(i, j) = value;
                result(i, j + x.numCols()) = squared;
                result(i, j + 2 * x.numCols()) = cubed;
                result(i, j + 3 * x.numCols()) = fourth;
                result(i, j + 4 * x.numCols()) = fourth * value;
            }
        }
    });
    return result;
}

Matrix DataAugmentor::random_fourier_features(const Matrix &x, int D, double gamma) {
    if (D <= 0) {
        throw std::invalid_argument("random_fourier_features: D must be positive.");
    }
    if (!std::isfinite(gamma) || gamma <= 0.0) {
        throw std::invalid_argument("random_fourier_features: gamma must be finite and positive.");
    }

    size_t n = x.numRows();   // Number of data points
    size_t d = x.numCols();   // Input dimension
    const size_t feature_count = static_cast<size_t>(D);

    Matrix result(n, feature_count);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal_dist(0, std::sqrt(2 * gamma));
    std::uniform_real_distribution<> uniform_dist(0, 2 * M_PI);

    // Create W: d × D
    Matrix W(d, feature_count);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < feature_count; ++j) {
            W(i, j) = normal_dist(gen);
        }
    }

    // Create b: 1 × D
    Matrix b(1, feature_count);
    for (size_t j = 0; j < feature_count; ++j) {
        b(0, j) = uniform_dist(gen);
    }

    // Compute RFFs
    double scale = std::sqrt(2.0 / static_cast<double>(feature_count));

    ctorch::parallel::parallel_for_items(n, d * feature_count, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            for (size_t j = 0; j < feature_count; ++j) {
                double dot_product = 0.0;
                for (size_t k = 0; k < d; ++k) {
                    dot_product += x(i, k) * W(k, j);
                }
                result(i, j) = scale * std::cos(dot_product + b(0, j));
            }
        }
    });

    return result;
}


Matrix DataAugmentor::augment_data(const Matrix &x, DataAugmentationType augmentation_type) {
    if (augmentation_type == DataAugmentationType::NO_OP) {
        return x;
    }
    else if (augmentation_type == DataAugmentationType::POLY_2) {
        return poly_2(x);
    }
    else if (augmentation_type == DataAugmentationType::POLY_3) {
        return poly_3(x);
    }
    else if (augmentation_type == DataAugmentationType::POLY_4) {
        return poly_4(x);
    }
    else if (augmentation_type == DataAugmentationType::POLY_5) {
        return poly_5(x);
    }
    else if (augmentation_type == DataAugmentationType::RFF) {
        return random_fourier_features(x, 100, 0.75);
    }

    throw std::invalid_argument("Unknown DataAugmentationType");
}
