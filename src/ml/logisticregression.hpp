#include "../math/ast.hpp"
#include "../math/matrix.hpp"
#include "../math/optim.hpp"
#include "../math/dataaugmentor.hpp"

using math::ASTNode;

namespace ml {
    class LogisticRegression {
        private:
            Matrix xTr;
            Matrix yTr;
            std::vector<Matrix> xTr_rows;
            Matrix weights;
            double bias;
            math::OptimParams optim_params;
            std::shared_ptr<ASTNode> single_loss(const Matrix &x, int y) const;
            std::shared_ptr<ASTNode> loss() const;
            DataAugmentationType data_augmentation_type;
        
        public:
            LogisticRegression(Matrix xTr, Matrix yTr, math::OptimParams optim_params, DataAugmentationType data_augmentation_type = DataAugmentationType::NO_OP);
            double predict(const Matrix &x) const;
            double score(const Matrix &xTe, const Matrix &yTe) const;
    };
}