#include "randomfouriersvm.hpp"
#include "distributed/distributed_optimizer.hpp"
#include "math/matrix.hpp"
#include "math/dataaugmentor.hpp"
#include "math/parallel.hpp"
#include <random>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "svm.hpp"

namespace ml
{
    RandomFourierSVM RandomFourierSVM::train_distributed(
        Matrix xTr,
        Matrix yTr,
        int D,
        double gamma,
        double learning_rate,
        int max_iter,
        double C,
        ctorch::distributed::ProcessGroup &group)
    {
        if (D <= 0)
        {
            throw std::invalid_argument("RandomFourierSVM: D must be positive.");
        }
        if (!std::isfinite(gamma) || gamma <= 0.0)
        {
            throw std::invalid_argument("RandomFourierSVM: gamma must be finite and positive.");
        }
        if (yTr.numRows() != 1)
        {
            throw std::invalid_argument("RandomFourierSVM: yTr must be a row vector (1 x N).");
        }
        if (xTr.numRows() != yTr.numCols())
        {
            throw std::invalid_argument("RandomFourierSVM: number of samples and labels must match.");
        }

        RandomFourierSVM model;
        model.D = D;
        model.gamma = gamma;
        model.W = Matrix(D, xTr.numCols());
        model.b = Matrix(D, 1);

        std::random_device rd;
        std::mt19937 gen(rd());
        const double w_std = std::sqrt(2.0 * gamma + 1e-15);
        std::normal_distribution<double> w_dist(0.0, w_std);
        std::uniform_real_distribution<double> b_dist(0.0, 2.0 * M_PI);

        if (group.rank() == 0)
        {
            for (int i = 0; i < D; ++i)
            {
                for (size_t j = 0; j < xTr.numCols(); ++j)
                {
                    model.W(i, j) = w_dist(gen);
                }
                model.b(i, 0) = b_dist(gen);
            }
        }

        group.broadcast(model.W, 0);
        group.broadcast(model.b, 0);

        Matrix transformed_xTr = model.transform_data(xTr);
        model.svm = SVM::train_distributed(
            transformed_xTr,
            yTr,
            learning_rate,
            max_iter,
            C,
            DataAugmentationType::NO_OP,
            group);
        return model;
    }

    RandomFourierSVM::RandomFourierSVM(Matrix xTr, Matrix yTr, int D, double gamma, double learning_rate, int max_iter, double C)
    {
        if (D <= 0)
        {
            throw std::invalid_argument("RandomFourierSVM: D must be positive.");
        }
        if (!std::isfinite(gamma) || gamma <= 0.0)
        {
            throw std::invalid_argument("RandomFourierSVM: gamma must be finite and positive.");
        }
        if (yTr.numRows() != 1)
        {
            throw std::invalid_argument("RandomFourierSVM: yTr must be a row vector (1 x N).");
        }
        if (xTr.numRows() != yTr.numCols())
        {
            throw std::invalid_argument("RandomFourierSVM: number of samples and labels must match.");
        }

        // Generate random Fourier features

        this->D = D;
        this->gamma = gamma;

        // Initialize W and b with random values
        W = Matrix(D, xTr.numCols());
        b = Matrix(D, 1);

        std::random_device rd;
        std::mt19937 gen(rd());
        const double w_std = std::sqrt(2.0 * gamma + 1e-15);
        std::normal_distribution<double> w_dist(0.0, w_std);
        std::uniform_real_distribution<double> b_dist(0.0, 2.0 * M_PI);

        for (int i = 0; i < D; ++i)
        {
            for (size_t j = 0; j < xTr.numCols(); ++j)
            {
                W(i, j) = w_dist(gen);
            }
            b(i, 0) = b_dist(gen);
        }

        // Transform the training data
        Matrix transformed_xTr = transform_data(xTr);

        // Train the SVM on the transformed data

        svm = SVM(transformed_xTr, yTr, learning_rate, max_iter, C, DataAugmentationType::NO_OP);
    }

    Matrix RandomFourierSVM::transform_data(const Matrix &x) const
    {
        if (x.numCols() != W.numCols())
        {
            throw std::invalid_argument("RandomFourierSVM: feature dimension mismatch.");
        }

        Matrix result(x.numRows(), D);
        ctorch::parallel::parallel_for_items(x.numRows(), x.numCols() * static_cast<size_t>(D), [&](size_t begin, size_t end)
        {
            for (size_t i = begin; i < end; ++i)
            {
                for (int j = 0; j < D; ++j)
                {
                    double dot_product = 0.0;
                    for (size_t k = 0; k < x.numCols(); ++k)
                    {
                        dot_product += x(i, k) * W(j, k);
                    }
                    result(i, j) = std::cos(dot_product + b(j, 0));
                }
            }
        });
        return result;
    }

    int RandomFourierSVM::predict(const Matrix &x) const
    {
        Matrix transformed_x = transform_data(x);
        return svm.predict(transformed_x);
    }

    double RandomFourierSVM::score(const Matrix &xTe, const Matrix &yTe) const
    {
        if (yTe.numRows() != 1)
        {
            throw std::invalid_argument("RandomFourierSVM::score: yTe must be a row vector (1 x N).");
        }
        if (xTe.numRows() != yTe.numCols())
        {
            throw std::invalid_argument("RandomFourierSVM::score: number of samples and labels must match.");
        }
        if (xTe.numRows() == 0)
        {
            throw std::invalid_argument("RandomFourierSVM::score: xTe must contain at least one sample.");
        }

        const size_t total = xTe.numRows();
        const int correct = ctorch::parallel::parallel_reduce_items<int>(
            total,
            xTe.numCols(),
            0,
            [&](size_t begin, size_t end)
            {
                int local_correct = 0;
                for (size_t i = begin; i < end; ++i)
                {
                    Matrix x_row(1, xTe.numCols());
                    for (size_t j = 0; j < xTe.numCols(); ++j)
                    {
                        x_row(0, j) = xTe(i, j);
                    }

                    const int prediction = predict(x_row);
                    const int true_label = static_cast<int>(yTe.at(0, i));

                    if (prediction == true_label)
                    {
                        ++local_correct;
                    }
                }
                return local_correct;
            },
            [](int lhs, int rhs)
            {
                return lhs + rhs;
            });

        return static_cast<double>(correct) / static_cast<double>(total);
    }
} 
