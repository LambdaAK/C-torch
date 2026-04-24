#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "distributed/distributed_optimizer.hpp"
#include "distributed/tcp_process_group.hpp"
#include "math/ast.hpp"
#include "math/matrix.hpp"
#include "math/optim.hpp"

namespace
{
struct ProgramOptions
{
    int rank = 0;
    int world_size = 1;
    std::string master_address = "127.0.0.1";
    std::uint16_t master_port = 29500;
    double learning_rate = 0.2;
    int max_iter = 200;
};

void print_usage(const char *program_name)
{
    std::cout << "Usage: " << program_name << " [options]\n"
              << "  --rank <n>         Worker rank (default: 0)\n"
              << "  --world-size <n>   Total number of workers (default: 1)\n"
              << "  --master-addr <a>  TCP master address (default: 127.0.0.1)\n"
              << "  --master-port <p>  TCP master port (default: 29500)\n"
              << "  --learning-rate <r> Learning rate (default: 0.2)\n"
              << "  --max-iter <n>     Training iterations (default: 200)\n";
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
        if (arg == "--learning-rate")
        {
            options.learning_rate = std::stod(require_value("--learning-rate"));
            continue;
        }
        if (arg == "--max-iter")
        {
            options.max_iter = std::stoi(require_value("--max-iter"));
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
    if (options.learning_rate <= 0.0)
    {
        throw std::invalid_argument("--learning-rate must be positive.");
    }
    if (options.max_iter <= 0)
    {
        throw std::invalid_argument("--max-iter must be positive.");
    }

    return options;
}

Matrix make_features()
{
    constexpr std::size_t grid_size = 8;
    constexpr std::size_t sample_count = grid_size * grid_size;
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

Matrix make_labels()
{
    constexpr std::size_t grid_size = 8;
    constexpr std::size_t sample_count = grid_size * grid_size;
    Matrix y(1, sample_count);

    std::size_t index = 0;
    for (std::size_t row = 0; row < grid_size; ++row)
    {
        const double x0 = -1.5 + 3.0 * static_cast<double>(row) / static_cast<double>(grid_size - 1);
        for (std::size_t col = 0; col < grid_size; ++col)
        {
            const double x1 = -1.5 + 3.0 * static_cast<double>(col) / static_cast<double>(grid_size - 1);
            y(0, index) = (x0 + 0.75 * x1 > 0.1) ? 1.0 : 0.0;
            ++index;
        }
    }

    return y;
}

class LogisticLoss final : public LossFunction
{
public:
    std::shared_ptr<math::ASTNode> sample_loss(const Matrix &x, double y) const override
    {
        std::shared_ptr<math::ASTNode> w_transpose_x = math::Num(0);
        for (std::size_t i = 0; i < x.numCols(); ++i)
        {
            w_transpose_x = w_transpose_x + math::Var("w" + std::to_string(i)) * math::Num(x.at(0, i));
        }

        const std::shared_ptr<math::ASTNode> b = math::Var("b");
        const std::shared_ptr<math::ASTNode> logits = w_transpose_x + b;
        const std::shared_ptr<math::ASTNode> y_hat = math::Sigmoid(logits);
        return -math::Num(y) * math::Log(y_hat) -
               (math::Num(1.0) - math::Num(y)) * math::Log(math::Num(1.0) - y_hat);
    }

    std::shared_ptr<math::ASTNode> regularizer() const override
    {
        return math::Num(0.0);
    }
};

double predict_probability(const std::unordered_map<std::string, double> &theta, double x0, double x1)
{
    const double logits = theta.at("w0") * x0 + theta.at("w1") * x1 + theta.at("b");
    return 1.0 / (1.0 + std::exp(-logits));
}

double accuracy(const std::unordered_map<std::string, double> &theta, const Matrix &x, const Matrix &y)
{
    std::size_t correct = 0;
    for (std::size_t i = 0; i < x.numRows(); ++i)
    {
        const double prob = predict_probability(theta, x(i, 0), x(i, 1));
        const int predicted = prob >= 0.5 ? 1 : 0;
        if (predicted == static_cast<int>(y(0, i)))
        {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(x.numRows());
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        const ProgramOptions options = parse_options(argc, argv);
        const Matrix x = make_features();
        const Matrix y = make_labels();
        const std::shared_ptr<LossFunction> loss_function = std::make_shared<LogisticLoss>();
        const math::OptimParams optim_params(
            math::OptimType::GD,
            options.learning_rate,
            options.max_iter,
            Matrix(),
            Matrix(),
            1);

        std::unordered_map<std::string, double> theta = {
            {"w0", 0.0},
            {"w1", 0.0},
            {"b", 0.0},
        };

        if (options.world_size == 1)
        {
            math::Optimizer optimizer(optim_params);
            theta = optimizer.optimize(loss_function, x, y, theta);
        }
        else
        {
            ctorch::distributed::TcpProcessGroup group(
                options.master_address,
                options.master_port,
                options.rank,
                options.world_size);
            ctorch::distributed::DistributedOptimizer optimizer(group, optim_params);
            theta = optimizer.optimize(loss_function, x, y, theta);
        }

        if (options.rank == 0)
        {
            std::cout << "learned logistic regression parameters:\n";
            std::cout << "  w0 = " << theta["w0"] << '\n';
            std::cout << "  w1 = " << theta["w1"] << '\n';
            std::cout << "   b = " << theta["b"] << '\n';
            std::cout << "training accuracy = " << accuracy(theta, x, y) << '\n';
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "distributed_logistic_regression: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
