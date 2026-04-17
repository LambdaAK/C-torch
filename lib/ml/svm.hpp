#pragma once
#include "math/matrix.hpp"
#include "math/dataaugmentor.hpp"

namespace ml {
    class SVM {
        private:
            Matrix xTr;
            Matrix yTr;
            Matrix weights;
            double bias;
            DataAugmentationType augmentation_type;

        public:
            SVM();
            SVM(Matrix xTr, Matrix yTr, double learning_rate, int max_iter, double C, DataAugmentationType augmentation_type);
            int predict(const Matrix &x) const;
            double score(const Matrix &xTe, const Matrix &yTe) const;
    };
}
