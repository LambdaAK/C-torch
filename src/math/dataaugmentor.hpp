#pragma once

#include "matrix.hpp"

enum class DataAugmentationType {
    NO_OP,
    POLY_2,
    POLY_3,
    POLY_4,
    POLY_5,
    RFF
};

class DataAugmentor {
    public:
    static Matrix no_op(const Matrix &x);
    static Matrix poly_2(const Matrix &x);
    static Matrix poly_3(const Matrix &x);
    static Matrix poly_4(const Matrix &x);
    static Matrix poly_5(const Matrix &x);
    static Matrix random_fourier_features(const Matrix &x, int D, double gamma);
    static Matrix augment_data(const Matrix &x, DataAugmentationType augmentation_type);
};