#include "kernelsvm.hpp"
#include "distributed/distributed_optimizer.hpp"
#include "math/optim.hpp"

using math::ASTNode;
using math::Max;
using math::Num;
using math::Var;

namespace
{
class KernelSVMLossFunction final : public LossFunction
{
private:
    std::vector<Matrix> xTr_rows_;
    Matrix kernel_matrix_;
    ml::KernelOptions kernel_options_;
    double C_;

public:
    KernelSVMLossFunction(Matrix xTr, ml::KernelOptions kernel_options, double C)
        : xTr_rows_(xTr.rowsAsMatrices()),
          kernel_matrix_(xTr.numRows(), xTr.numRows()),
          kernel_options_(kernel_options),
          C_(C)
    {
        for (std::size_t i = 0; i < xTr_rows_.size(); ++i)
        {
            for (std::size_t j = 0; j < xTr_rows_.size(); ++j)
            {
                kernel_matrix_(i, j) = ml::Kernels::kernel(xTr_rows_[i], xTr_rows_[j], kernel_options_);
            }
        }
    }

    std::shared_ptr<ASTNode> sample_loss(const Matrix &x, double y) const override
    {
        std::shared_ptr<ASTNode> inside = Num(0);
        for (std::size_t i = 0; i < xTr_rows_.size(); ++i)
        {
            std::shared_ptr<ASTNode> alpha_i = Var("alpha" + std::to_string(i));
            inside = inside + alpha_i * Num(ml::Kernels::kernel(xTr_rows_[i], x, kernel_options_));
        }

        inside = inside + Var("b");
        inside = Num(y) * inside;
        return Num(C_) * Max(Num(0), Num(1) - inside);
    }

    std::shared_ptr<ASTNode> regularizer() const override
    {
        std::shared_ptr<ASTNode> reg = Num(0);
        for (std::size_t i = 0; i < xTr_rows_.size(); ++i)
        {
            for (std::size_t j = 0; j < xTr_rows_.size(); ++j)
            {
                std::shared_ptr<ASTNode> alpha_i = Var("alpha" + std::to_string(i));
                std::shared_ptr<ASTNode> alpha_j = Var("alpha" + std::to_string(j));
                reg = reg + alpha_i * alpha_j * Num(kernel_matrix_(i, j));
            }
        }
        return reg / Num(2);
    }
};
} // namespace

namespace ml
{
    KernelSVM KernelSVM::train_distributed(
        Matrix xTr,
        Matrix yTr,
        double learning_rate,
        int max_iter,
        double C,
        KernelOptions kernel_options,
        ctorch::distributed::ProcessGroup &group)
    {
        KernelSVM model;
        model.xTr = xTr;
        model.yTr = yTr;
        model.learning_rate = learning_rate;
        model.max_iter = max_iter;
        model.C = C;
        model.kernel_options = kernel_options;
        model.xTr_rows = xTr.rowsAsMatrices();
        model.weights = Matrix(1, xTr.numRows());
        model.bias = 0.0;

        if (model.xTr_rows.size() < 2)
        {
            throw std::invalid_argument("At least two training points are required.");
        }

        std::shared_ptr<LossFunction> loss_function = std::make_shared<KernelSVMLossFunction>(xTr, kernel_options, C);

        std::unordered_map<std::string, double> values;
        for (size_t i = 0; i < xTr.numRows(); i++)
        {
            std::string w_i = "alpha" + std::to_string(i);
            values[w_i] = 1;
        }
        values["b"] = 1;

        math::OptimParams optim_params(math::OptimType::GD, learning_rate, max_iter);
        ctorch::distributed::DistributedOptimizer optimizer(group, optim_params);
        std::unordered_map<std::string, double> result = optimizer.optimize(loss_function, model.xTr, model.yTr, values);

        for (size_t i = 0; i < xTr.numRows(); i++)
        {
            std::string w_i = "alpha" + std::to_string(i);
            model.weights(0, i) = result[w_i];
        }

        model.bias = result["b"];
        return model;
    }

    KernelSVM::KernelSVM(Matrix xTr, Matrix yTr, double learning_rate, int max_iter, double C, KernelOptions kernel_options)
        : xTr(xTr), yTr(yTr), learning_rate(learning_rate), max_iter(max_iter), C(C), kernel_options(kernel_options)
    {
        // Initialize weights and bias
        weights = Matrix(1, xTr.numRows());
        bias = 0;

        xTr_rows = xTr.rowsAsMatrices();

        // Check that we have at least two training points
        if (xTr_rows.size() < 2)
        {
            throw std::invalid_argument("At least two training points are required.");
        }

        // compute the loss function

        std::shared_ptr<ASTNode> loss_function = this->loss_function();

        // optimize the loss function with respect to the weights and bias

        std::unordered_map<std::string, double> values;

        for (size_t i = 0; i < xTr.numRows(); i++)
        {
            std::string w_i = "alpha" + std::to_string(i);
            values[w_i] = 1;
        }

        values["b"] = 1;

        // optimize the loss function with GD

        math::GD gd(learning_rate, max_iter);

        std::unordered_map<std::string, double> result = gd.optimize(loss_function, values);

        // Update weights and bias with the optimized values
        for (size_t i = 0; i < xTr.numRows(); i++)
        {
            std::string w_i = "alpha" + std::to_string(i);
            weights(0, i) = result[w_i];
        }

        bias = result["b"];

        // finished training

        // print the parameters

        weights.print();

        std::cout << bias << std::endl;
    }

    std::shared_ptr<math::ASTNode> KernelSVM::loss_function()
    {
        // parameters alpha_1, ...., \alpha_n, b

        // first, construct the regularizer term

        std::shared_ptr<ASTNode> reg = Num(0);

        // compute the kernel matrix for better efficiency

        Matrix kernel_matrix(xTr.numRows(), xTr.numRows());

        for (size_t i = 0; i < xTr.numRows(); i++)
        {
            for (size_t j = 0; j < xTr.numRows(); j++)
            {
                kernel_matrix(i, j) = Kernels::kernel(xTr_rows[i], xTr_rows[j], kernel_options);
            }
        }

        for (size_t i = 0; i < weights.numCols(); i++)
        {
            for (size_t j = 0; j < weights.numCols(); j++)
            {
                std::shared_ptr<ASTNode> alpha_i = Var("alpha" + std::to_string(i));
                std::shared_ptr<ASTNode> alpha_j = Var("alpha" + std::to_string(j));

                // print the rows

                Matrix x_i = xTr_rows.at(i);

                Matrix x_j = xTr_rows.at(j);

                reg = reg + alpha_i * alpha_j * Num(kernel_matrix(i, j));
            }
        }

        // multiply by 1/2

        reg = reg / Num(2);

        // next, compute the hinge losses

        std::shared_ptr<ASTNode> hinge_losses = Num(0);

        // for each example
        for (size_t k = 0; k < xTr.numRows(); k++)
        {
            // max of 0 and another term

            // compute y_k * sum_{i = 1} ^ n \alpha_i * K(x_i, x_k) + b
            // this term is inside

            std::shared_ptr<ASTNode> inside = Num(0);

            double y_k = yTr.at(0, k);

            for (size_t i = 0; i < xTr.numRows(); i++)
            {

                Matrix x_i = xTr_rows.at(i);
                Matrix x_k = xTr_rows.at(k);
                std::shared_ptr<ASTNode> alpha_i = Var("alpha" + std::to_string(i));
                inside = inside + alpha_i * Num(kernel_matrix(i, k));
            }

            // the sum is done, now add the bias

            std::shared_ptr<ASTNode> b = Var("b");

            inside = inside + b;

            // multiply by y_k

            inside = Num(y_k) * inside;

            // compute the hinge loss, which is max(0, 1 - inside)

            std::shared_ptr<ASTNode> hinge_loss = Max(Num(0), Num(1) - inside);
            hinge_losses = hinge_losses + hinge_loss;
        }

        // multiply the hinge losses by C

        hinge_losses = Num(C) * hinge_losses;

        // add them to the regularizer

        std::shared_ptr<ASTNode> loss = reg + hinge_losses;

        return loss;
    }

    int KernelSVM::predict(const Matrix &x) const
    {
        // use the kernel to predict
        /*
            First, compute

            \sum_{i = 1} ^ n \alpha_i K(x_i, x)

        */

        double s = 0.0;

        for (size_t i = 0; i < xTr_rows.size(); i++)
        {
            s += weights(0, i) * Kernels::kernel(xTr_rows[i], x, kernel_options);
        }

        // add the bias

        s += bias;

        // take the sign

        if (s >= 0)
        {
            return 1;
        }

        else
        {
            return -1;
        }
    }

    double KernelSVM::score(const Matrix &xTe, const Matrix &yTe) const
    {
        std::vector<Matrix> xTe_rows = xTe.rowsAsMatrices();

        int correct = 0;
        int total = xTe.numRows();

        for (size_t i = 0; i < total; i++)
        {
            int prediction = predict(xTe_rows[i]);

            // Get the true label
            int true_label = static_cast<int>(yTe.at(0, i));

            if (prediction == true_label)
            {
                correct++;
            }
        }

        return static_cast<double>(correct) / total;
    }
}
