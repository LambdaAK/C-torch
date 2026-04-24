#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "distributed/tcp_process_group.hpp"
#include "math/dataaugmentor.hpp"
#include "math/matrix.hpp"
#include "math/optim.hpp"
#include "ml/kernelsvm.hpp"
#include "ml/linearregression.hpp"
#include "ml/logisticregression.hpp"
#include "ml/randomfouriersvm.hpp"
#include "ml/svm.hpp"

namespace
{
struct ProgramOptions
{
    int rank = 0;
    int world_size = 1;
    std::string master_address = "127.0.0.1";
    std::uint16_t master_port = 29500;
};

void print_usage(const char *program_name)
{
    std::cout << "Usage: " << program_name << " [options]\n"
              << "  --rank <n>         Worker rank (default: 0)\n"
              << "  --world-size <n>   Number of workers (default: 1)\n"
              << "  --master-addr <a>  TCP master address (default: 127.0.0.1)\n"
              << "  --master-port <p>  TCP master port (default: 29500)\n";
}

ProgramOptions parse_options(int argc, char **argv)
{
    ProgramOptions options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto require_value = [&](const char *name) -> std::string
        {
            if (i + 1 >= argc)
            {
                throw std::invalid_argument(std::string(name) + " requires a value.");
            }
            return argv[++i];
        };

        if (arg == "--rank")
        {
            options.rank = std::stoi(require_value("--rank"));
            continue;
        }
        if (arg == "--world-size")
        {
            options.world_size = std::stoi(require_value("--world-size"));
            continue;
        }
        if (arg == "--master-addr")
        {
            options.master_address = require_value("--master-addr");
            continue;
        }
        if (arg == "--master-port")
        {
            options.master_port = static_cast<std::uint16_t>(std::stoul(require_value("--master-port")));
            continue;
        }
        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            std::exit(0);
        }

        throw std::invalid_argument("Unknown argument: " + arg);
    }

    if (options.rank < 0)
    {
        throw std::invalid_argument("--rank must be non-negative.");
    }
    if (options.world_size <= 0)
    {
        throw std::invalid_argument("--world-size must be positive.");
    }
    if (options.rank >= options.world_size)
    {
        throw std::invalid_argument("--rank must be less than --world-size.");
    }

    return options;
}

Matrix make_grid_features(std::size_t grid_size)
{
    const std::size_t sample_count = grid_size * grid_size;
    Matrix x(sample_count, 2);
    std::size_t index = 0;
    for (std::size_t row = 0; row < grid_size; ++row)
    {
        const double x0 = -1.5 + 3.0 * static_cast<double>(row) / static_cast<double>(grid_size - 1);
        for (std::size_t col = 0; col < grid_size; ++col)
        {
            const double x1 = -1.5 + 3.0 * static_cast<double>(col) / static_cast<double>(grid_size - 1);
            x(index, 0) = x0;
            x(index, 1) = x1;
            ++index;
        }
    }
    return x;
}

Matrix make_linear_targets(const Matrix &x)
{
    Matrix y(1, x.numRows());
    for (std::size_t i = 0; i < x.numRows(); ++i)
    {
        const double x0 = x(i, 0);
        const double x1 = x(i, 1);
        y(0, i) = 2.0 * x0 - 3.0 * x1 + 0.5;
    }
    return y;
}

Matrix make_logistic_labels(const Matrix &x)
{
    Matrix y(1, x.numRows());
    for (std::size_t i = 0; i < x.numRows(); ++i)
    {
        const double score = x(i, 0) + 0.75 * x(i, 1);
        y(0, i) = score > 0.1 ? 1.0 : 0.0;
    }
    return y;
}

Matrix make_signed_labels(const Matrix &x)
{
    Matrix y(1, x.numRows());
    for (std::size_t i = 0; i < x.numRows(); ++i)
    {
        const double score = x(i, 0) + 0.75 * x(i, 1);
        y(0, i) = score > 0.1 ? 1.0 : -1.0;
    }
    return y;
}

Matrix make_nonlinear_labels(const Matrix &x)
{
    Matrix y(1, x.numRows());
    for (std::size_t i = 0; i < x.numRows(); ++i)
    {
        const double radius_sq = x(i, 0) * x(i, 0) + x(i, 1) * x(i, 1);
        y(0, i) = radius_sq < 1.0 ? 1.0 : -1.0;
    }
    return y;
}

double mean_squared_error(const ml::LinearRegression &model, const Matrix &x, const Matrix &y)
{
    double total = 0.0;
    for (std::size_t i = 0; i < x.numRows(); ++i)
    {
        Matrix row(1, x.numCols());
        for (std::size_t j = 0; j < x.numCols(); ++j)
        {
            row(0, j) = x(i, j);
        }
        const double predicted = model.predict(row);
        const double error = predicted - y(0, i);
        total += error * error;
    }
    return total / static_cast<double>(x.numRows());
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        const ProgramOptions options = parse_options(argc, argv);
        const std::size_t grid_size = 10;
        const Matrix x = make_grid_features(grid_size);
        const Matrix y_linear = make_linear_targets(x);
        const Matrix y_logistic = make_logistic_labels(x);
        const Matrix y_signed = make_signed_labels(x);
        const Matrix y_nonlinear = make_nonlinear_labels(x);

        std::unique_ptr<ctorch::distributed::TcpProcessGroup> group;
        if (options.world_size > 1)
        {
            group = std::make_unique<ctorch::distributed::TcpProcessGroup>(
                options.master_address,
                options.master_port,
                options.rank,
                options.world_size);
        }

        if (options.rank == 0)
        {
            std::cout << "Distributed classical model demo\n";
        }

        if (group)
        {
            ml::LinearRegression linear = ml::LinearRegression::train_distributed(
                x,
                y_linear,
                0.02,
                1200,
                *group);
            math::OptimParams logistic_params(
                math::OptimType::GD,
                0.05,
                800,
                Matrix(),
                Matrix(),
                1);
            ml::LogisticRegression logistic = ml::LogisticRegression::train_distributed(
                x,
                y_logistic,
                logistic_params,
                DataAugmentationType::NO_OP,
                *group);
            ml::SVM svm = ml::SVM::train_distributed(
                x,
                y_signed,
                0.05,
                800,
                1.0,
                DataAugmentationType::NO_OP,
                *group);
            ml::KernelSVM kernel_svm = ml::KernelSVM::train_distributed(
                x,
                y_nonlinear,
                0.03,
                500,
                1.0,
                ml::KernelOptions::radial_basis(1.0),
                *group);
            ml::RandomFourierSVM rff = ml::RandomFourierSVM::train_distributed(
                x,
                y_nonlinear,
                16,
                1.0,
                0.03,
                500,
                1.0,
                *group);

            if (options.rank == 0)
            {
                std::cout << "Linear regression MSE: " << mean_squared_error(linear, x, y_linear) << '\n';
                std::cout << "Logistic regression accuracy: " << logistic.score(x, y_logistic) << '\n';
                std::cout << "Linear SVM accuracy: " << svm.score(x, y_signed) << '\n';
                std::cout << "Kernel SVM accuracy: " << kernel_svm.score(x, y_nonlinear) << '\n';
                std::cout << "Random Fourier SVM accuracy: " << rff.score(x, y_nonlinear) << '\n';
            }
        }
        else
        {
            ml::LinearRegression linear(x, y_linear, 0.02, 1200);
            math::OptimParams logistic_params(
                math::OptimType::GD,
                0.05,
                800,
                Matrix(),
                Matrix(),
                1);
            ml::LogisticRegression logistic(
                x,
                y_logistic,
                logistic_params,
                DataAugmentationType::NO_OP);
            ml::SVM svm(x, y_signed, 0.05, 800, 1.0, DataAugmentationType::NO_OP);
            ml::KernelSVM kernel_svm(x, y_nonlinear, 0.03, 500, 1.0, ml::KernelOptions::radial_basis(1.0));
            ml::RandomFourierSVM rff(x, y_nonlinear, 16, 1.0, 0.03, 500, 1.0);

            std::cout << "Linear regression MSE: " << mean_squared_error(linear, x, y_linear) << '\n';
            std::cout << "Logistic regression accuracy: " << logistic.score(x, y_logistic) << '\n';
            std::cout << "Linear SVM accuracy: " << svm.score(x, y_signed) << '\n';
            std::cout << "Kernel SVM accuracy: " << kernel_svm.score(x, y_nonlinear) << '\n';
            std::cout << "Random Fourier SVM accuracy: " << rff.score(x, y_nonlinear) << '\n';
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "distributed_classical_models: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
