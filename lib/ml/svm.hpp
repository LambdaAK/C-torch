#pragma once
#include "math/matrix.hpp"
#include "math/ast.hpp"
#include "math/dataaugmentor.hpp"

using math::ASTNode;

namespace ml {
    class SVM {
        private:
            Matrix xTr;
            Matrix yTr;
            std::vector<Matrix> xTr_rows;
            Matrix weights;
            double bias;
            double learning_rate;
            int max_iter;
            double C;
            DataAugmentationType augmentation_type;
            std::shared_ptr<ASTNode> single_sample_loss(const Matrix &x, int y) const;
            std::shared_ptr<ASTNode> loss_function() const;

        public:
            SVM();
            SVM(Matrix xTr, Matrix yTr, double learning_rate, int max_iter, double C, DataAugmentationType augmentation_type);
            int predict(const Matrix &x) const;
            double score(const Matrix &xTe, const Matrix &yTe) const;
    };
}