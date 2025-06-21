#include "../math/matrix.hpp"
#include "../math/ast.hpp"

using math::ASTNode;

namespace ml
{
    class LinearRegression
    {
    private:
        Matrix xTr;
        std::vector<Matrix> xTr_rows;
        Matrix yTr;
        Matrix weights;
        double bias;
        double learning_rate;
        int max_iter;

        std::shared_ptr<ASTNode> single_squared_loss(const Matrix &x, double y) const;
        std::shared_ptr<ASTNode> OLS_loss() const;

    public:
        LinearRegression(Matrix xTr, Matrix yTr, double learning_rate, int max_iter);
        double predict(const Matrix &x) const;
        double score(const Matrix &xTe, const Matrix &yTe, double threshold = 0.5) const;
    };
}