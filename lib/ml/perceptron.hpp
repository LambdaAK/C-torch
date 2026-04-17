#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

#include "math/matrix.hpp"

namespace ml
{
    class Perceptron
    {
    private:
        Matrix weights;
        double bias;

    public:
        Perceptron(Matrix xTr, Matrix yTr, int epochs = 300);
        int predict(const Matrix &x) const; // row vector
        double score(const Matrix &xTe, const Matrix &yTe) const;
    };
}
