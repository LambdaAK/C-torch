#include "dataaugmentor.hpp"
#include <random>
#include <cmath>

// Add this if M_PI is not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Matrix DataAugmentor::no_op(const Matrix &x) {
    return x;
}

Matrix DataAugmentor::poly_2(const Matrix &x) {
    Matrix result(x.numRows(), x.numCols() * 2);
    for (size_t i = 0; i < x.numRows(); ++i) {
        for (size_t j = 0; j < x.numCols(); ++j) {
            result(i, j) = x(i, j);
            result(i, j + x.numCols()) = x(i, j) * x(i, j);
        }
    }
    return result;
}

Matrix DataAugmentor::poly_3(const Matrix &x) {
    Matrix result(x.numRows(), x.numCols() * 3);
    for (size_t i = 0; i < x.numRows(); ++i) {
        for (size_t j = 0; j < x.numCols(); ++j) {
            // Original feature
            result(i, j) = x(i, j);
            
            // Squared feature (x²)
            result(i, j + x.numCols()) = x(i, j) * x(i, j);
            
            // Cubed feature (x³)
            result(i, j + 2 * x.numCols()) = x(i, j) * x(i, j) * x(i, j);
        }
    }
    return result;
}

Matrix DataAugmentor::poly_4(const Matrix &x) {
    Matrix result(x.numRows(), x.numCols() * 4);
    for (size_t i = 0; i < x.numRows(); ++i) {
        for (size_t j = 0; j < x.numCols(); ++j) {
            // Original feature (x)
            result(i, j) = x(i, j);
            
            // Squared feature (x²)
            result(i, j + x.numCols()) = x(i, j) * x(i, j);
            
            // Cubed feature (x³)
            result(i, j + 2 * x.numCols()) = x(i, j) * x(i, j) * x(i, j);
            
            // Fourth power feature (x⁴)
            result(i, j + 3 * x.numCols()) = x(i, j) * x(i, j) * x(i, j) * x(i, j);
        }
    }
    return result;
}

Matrix DataAugmentor::poly_5(const Matrix &x) {
    Matrix result(x.numRows(), x.numCols() * 5);
    for (size_t i = 0; i < x.numRows(); ++i) {
        for (size_t j = 0; j < x.numCols(); ++j) {
            // Original feature (x)
            result(i, j) = x(i, j);
            
            // Squared feature (x²)
            result(i, j + x.numCols()) = x(i, j) * x(i, j);
            
            // Cubed feature (x³)
            result(i, j + 2 * x.numCols()) = x(i, j) * x(i, j) * x(i, j);
            
            // Fourth power feature (x⁴)
            result(i, j + 3 * x.numCols()) = x(i, j) * x(i, j) * x(i, j) * x(i, j);
            
            // Fifth power feature (x⁵)
            result(i, j + 4 * x.numCols()) = x(i, j) * x(i, j) * x(i, j) * x(i, j) * x(i, j);
        }
    }
    return result;
}

Matrix DataAugmentor::random_fourier_features(const Matrix &x, int D, double gamma) {
    size_t n = x.numRows();   // Number of data points
    size_t d = x.numCols();   // Input dimension

    Matrix result(n, D);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal_dist(0, std::sqrt(2 * gamma));
    std::uniform_real_distribution<> uniform_dist(0, 2 * M_PI);

    // Create W: d × D
    Matrix W(d, D);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < D; ++j) {
            W(i, j) = normal_dist(gen);
        }
    }

    // Create b: 1 × D
    Matrix b(1, D);
    for (size_t j = 0; j < D; ++j) {
        b(0, j) = uniform_dist(gen);
    }

    // Compute RFFs
    double scale = std::sqrt(2.0 / D);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < D; ++j) {
            double dot_product = 0.0;
            for (size_t k = 0; k < d; ++k) {
                dot_product += x(i, k) * W(k, j);
            }
            result(i, j) = scale * std::cos(dot_product + b(0, j));
        }
    }

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

}