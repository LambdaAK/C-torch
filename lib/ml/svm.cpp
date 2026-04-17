#include "svm.hpp"

#include "math/dataaugmentor.hpp"
#include "math/optim_qp.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ml
{
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
