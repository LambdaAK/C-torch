#include "svm.hpp"

#include "distributed/distributed_optimizer.hpp"
#include "math/dataaugmentor.hpp"
#include "math/ast.hpp"
#include "math/optim.hpp"
#include "math/optim_qp.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

using math::ASTNode;
using math::Max;
using math::Num;
using math::Var;

namespace
{
class LinearSVMHingeLossFunction final : public LossFunction
{
private:
    std::size_t feature_count_;
    double c_value_;

public:
    LinearSVMHingeLossFunction(std::size_t feature_count, double c_value)
        : feature_count_(feature_count), c_value_(c_value)
    {
    }

    std::shared_ptr<math::ASTNode> sample_loss(const Matrix &x, double y) const override
    {
        std::shared_ptr<ASTNode> score = Num(0);
        for (std::size_t i = 0; i < feature_count_; ++i)
        {
            score = score + Var("w" + std::to_string(i)) * Num(x.at(0, i));
        }
        score = score + Var("b");
        return Num(c_value_) * Max(Num(0), Num(1) - Num(y) * score);
    }

    std::shared_ptr<math::ASTNode> regularizer() const override
    {
        std::shared_ptr<ASTNode> reg = Num(0);
        for (std::size_t i = 0; i < feature_count_; ++i)
        {
            std::shared_ptr<ASTNode> w_i = Var("w" + std::to_string(i));
            reg = reg + (w_i ^ Num(2));
        }
        return Num(0.5) * reg;
    }
};
} // namespace

namespace ml
{
    SVM SVM::train_distributed(
        Matrix xTr,
        Matrix yTr,
        double learning_rate,
        int max_iter,
        double C,
        DataAugmentationType augmentation_type,
        ctorch::distributed::ProcessGroup &group)
    {
        if (xTr.numRows() != yTr.numCols())
        {
            throw std::invalid_argument("Number of training samples and labels must be equal.");
        }

        if (yTr.numRows() != 1)
        {
            throw std::invalid_argument("Labels must be a row vector.");
        }

        if (C <= 0)
        {
            throw std::invalid_argument("C must be positive.");
        }

        if (max_iter <= 0)
        {
            throw std::invalid_argument("max_iter must be positive.");
        }

        xTr = DataAugmentor::augment_data(xTr, augmentation_type);

        SVM model;
        model.xTr = xTr;
        model.yTr = yTr;
        model.augmentation_type = augmentation_type;
        model.weights = Matrix(1, xTr.numCols());
        model.bias = 0.0;

        std::shared_ptr<LossFunction> loss_function = std::make_shared<LinearSVMHingeLossFunction>(xTr.numCols(), C);

        std::unordered_map<std::string, double> values;
        for (std::size_t i = 0; i < xTr.numCols(); ++i)
        {
            values["w" + std::to_string(i)] = 0.0;
        }
        values["b"] = 0.0;

        math::OptimParams optim_params(math::OptimType::GD, learning_rate, max_iter);
        ctorch::distributed::DistributedOptimizer optimizer(group, optim_params);
        std::unordered_map<std::string, double> result = optimizer.optimize(loss_function, model.xTr, model.yTr, values);

        for (std::size_t i = 0; i < xTr.numCols(); ++i)
        {
            model.weights(0, i) = result["w" + std::to_string(i)];
        }
        model.bias = result["b"];
        return model;
    }

    SVM::SVM()
    {
        // default constructor
    }

    SVM::SVM(Matrix xTr, Matrix yTr, double learning_rate, int max_iter, double C, DataAugmentationType augmentation_type)
        : xTr(xTr),
          yTr(yTr),
          augmentation_type(augmentation_type)
    {
        if (xTr.numRows() != yTr.numCols())
        {
            throw std::invalid_argument("Number of training samples and labels must be equal.");
        }

        if (yTr.numRows() != 1)
        {
            throw std::invalid_argument("Labels must be a row vector.");
        }

        if (C <= 0)
        {
            throw std::invalid_argument("C must be positive.");
        }

        if (max_iter <= 0)
        {
            throw std::invalid_argument("max_iter must be positive.");
        }

        this->xTr = DataAugmentor::augment_data(xTr, augmentation_type);

        const size_t n_samples = this->xTr.numRows();
        const size_t n_features = this->xTr.numCols();

        std::vector<double> labels(n_samples, 0.0);
        for (size_t i = 0; i < n_samples; ++i)
        {
            labels[i] = this->yTr(0, i);
            if (std::abs(std::abs(labels[i]) - 1.0) > 1e-9)
            {
                throw std::invalid_argument("SVM labels must be {-1, +1}.");
            }
        }

        Matrix q(n_samples, n_samples);
        for (size_t i = 0; i < n_samples; ++i)
        {
            for (size_t j = 0; j < n_samples; ++j)
            {
                double x_dot = 0.0;
                for (size_t k = 0; k < n_features; ++k)
                {
                    x_dot += this->xTr(i, k) * this->xTr(j, k);
                }

                q(i, j) = labels[i] * labels[j] * x_dot;
            }
        }

        std::vector<double> c_vec(n_samples, -1.0);
        std::vector<double> lower_bounds(n_samples, 0.0);
        std::vector<double> upper_bounds(n_samples, C);

        // Equality constraint for SVM dual: sum_i alpha_i * y_i = 0.
        std::vector<double> equality_coeffs = labels;

        // Very small legacy learning rates (e.g. 1e-4) were tuned for hinge-loss GD,
        // so default to an auto step size for the QP solver unless a meaningful step is provided.
        const double qp_step_size = (learning_rate > 1e-3) ? learning_rate : 0.0;

        math::QuadraticProgramSolver qp_solver(
            max_iter,
            1e-6,
            50,
            qp_step_size);

        const math::QuadraticProgramResult qp_result = qp_solver.solve(
            q,
            c_vec,
            lower_bounds,
            upper_bounds,
            equality_coeffs,
            0.0);

        const std::vector<double> &alphas = qp_result.solution;

        weights = Matrix(1, n_features);
        for (size_t i = 0; i < n_samples; ++i)
        {
            const double alpha_y = alphas[i] * labels[i];
            for (size_t j = 0; j < n_features; ++j)
            {
                weights(0, j) += alpha_y * this->xTr(i, j);
            }
        }

        auto decision_without_bias = [&](size_t row_idx)
        {
            double value = 0.0;
            for (size_t j = 0; j < n_features; ++j)
            {
                value += weights(0, j) * this->xTr(row_idx, j);
            }
            return value;
        };

        const double alpha_tol = 1e-5;
        double bias_sum = 0.0;
        int bias_count = 0;

        for (size_t i = 0; i < n_samples; ++i)
        {
            if (alphas[i] > alpha_tol && alphas[i] < (C - alpha_tol))
            {
                bias_sum += labels[i] - decision_without_bias(i);
                ++bias_count;
            }
        }

        if (bias_count == 0)
        {
            for (size_t i = 0; i < n_samples; ++i)
            {
                if (alphas[i] > alpha_tol)
                {
                    bias_sum += labels[i] - decision_without_bias(i);
                    ++bias_count;
                }
            }
        }

        bias = (bias_count > 0) ? (bias_sum / static_cast<double>(bias_count)) : 0.0;
    }

    int SVM::predict(const Matrix &x) const
    {
        const Matrix x_augmented = DataAugmentor::augment_data(x, augmentation_type);

        if (x_augmented.numCols() != weights.numCols())
        {
            throw std::invalid_argument("Feature dimension mismatch between input and trained SVM weights.");
        }

        double score = weights.inner_product(x_augmented) + bias;
        return (score >= 0.0) ? 1 : -1;
    }

    double SVM::score(const Matrix &xTe, const Matrix &yTe) const
    {
        if (yTe.numRows() != 1)
        {
            throw std::invalid_argument("Labels must be a row vector (1 × N).");
        }

        if (xTe.numRows() != yTe.numCols())
        {
            throw std::invalid_argument("Mismatch between number of test samples and labels.");
        }

        int correct = 0;
        for (size_t i = 0; i < xTe.numRows(); ++i)
        {
            Matrix x_i(1, xTe.numCols());
            for (size_t j = 0; j < xTe.numCols(); ++j)
            {
                x_i(0, j) = xTe(i, j);
            }

            const int prediction = predict(x_i);
            const int actual = static_cast<int>(yTe(0, i));
            if (prediction == actual)
            {
                correct++;
            }
        }

        return static_cast<double>(correct) / xTe.numRows();
    }

} // namespace ml
