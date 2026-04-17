#pragma once
#include "math/matrix.hpp"
#include "svm.hpp"

namespace ml
{

    class RandomFourierSVM
    {
    private:
        int D;
        double gamma;
        Matrix W;
        Matrix b;
        SVM svm;

        Matrix transform_data(const Matrix &x) const;

    public:
        RandomFourierSVM(Matrix xTr, Matrix yTr, int D, double gamma, double learning_rate, int max_iter, double C);
        int predict(const Matrix &x) const;
        double score(const Matrix &xTe, const Matrix &yTe) const;
        ;
    };

}