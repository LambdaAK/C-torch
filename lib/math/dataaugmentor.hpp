#pragma once

#include "matrix.hpp"

/**
 * @brief Supported feature-augmentation modes.
 */
enum class DataAugmentationType {
    NO_OP,  ///< Return input unchanged.
    POLY_2, ///< Polynomial expansion up to degree 2.
    POLY_3, ///< Polynomial expansion up to degree 3.
    POLY_4, ///< Polynomial expansion up to degree 4.
    POLY_5, ///< Polynomial expansion up to degree 5.
    RFF     ///< Random Fourier feature expansion.
};

/**
 * @brief Stateless utilities for transforming feature matrices.
 */
class DataAugmentor {
    public:
    /**
     * @brief No-op transform.
     * @param x Input feature matrix.
     * @return Unchanged input matrix.
     */
    static Matrix no_op(const Matrix &x);

    /**
     * @brief Degree-2 polynomial augmentation.
     * @param x Input feature matrix.
     * @return Augmented feature matrix.
     */
    static Matrix poly_2(const Matrix &x);

    /**
     * @brief Degree-3 polynomial augmentation.
     * @param x Input feature matrix.
     * @return Augmented feature matrix.
     */
    static Matrix poly_3(const Matrix &x);

    /**
     * @brief Degree-4 polynomial augmentation.
     * @param x Input feature matrix.
     * @return Augmented feature matrix.
     */
    static Matrix poly_4(const Matrix &x);

    /**
     * @brief Degree-5 polynomial augmentation.
     * @param x Input feature matrix.
     * @return Augmented feature matrix.
     */
    static Matrix poly_5(const Matrix &x);

    /**
     * @brief Random Fourier feature mapping.
     * @param x Input feature matrix.
     * @param D Number of random Fourier components.
     * @param gamma RBF bandwidth parameter.
     * @return RFF-transformed matrix.
     */
    static Matrix random_fourier_features(const Matrix &x, int D, double gamma);

    /**
     * @brief Dispatches to augmentation strategy by enum.
     * @param x Input feature matrix.
     * @param augmentation_type Selected augmentation mode.
     * @return Transformed feature matrix.
     */
    static Matrix augment_data(const Matrix &x, DataAugmentationType augmentation_type);
};
