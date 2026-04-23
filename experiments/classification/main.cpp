#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <deque>
#include <fstream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <random>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <filesystem>
#include <cstdlib>
#include <stdexcept>

#include "math/matrix.hpp"
#include "math/ast.hpp"
#include "math/optim.hpp"
#include "ml/logisticregression.hpp"
#include "ml/svm.hpp"
#include "ml/linearregression.hpp"
#include "ml/kernelsvm.hpp"
#include "math/dataaugmentor.hpp"
#include "distributed/distributed_sync.hpp"
#include "distributed/distributed_checkpoint.hpp"
#include "distributed/distributed_training.hpp"
#include "distributed/tcp_process_group.hpp"
#include "ml/randomfouriersvm.hpp"
#include "ml/nn.hpp"
#include "ml/parallel_training.hpp"
#include "ml/perceptron.hpp"
#include "ml/knn.hpp"
#include "ml/gaussian_nb.hpp"

#include "../recommender/csv.h"

// log macro that prints the input
#define LOG(x) std::cout << x << std::endl

struct ProgramOptions
{
    bool distributed = false;
    int rank = 0;
    int world_size = 1;
    std::string master_address = "127.0.0.1";
    std::uint16_t master_port = 29500;
    std::size_t epochs = 1000;
    std::size_t batch_size = 64;
    std::uint64_t seed = 1337;
    std::string checkpoint_prefix;
    bool resume_checkpoint = false;
};

void print_usage(const char *program_name)
{
    std::cout << "Usage: " << program_name << " [options]\n"
              << "  --distributed           Enable synchronous data-parallel training\n"
              << "  --rank <n>              Worker rank (default: 0)\n"
              << "  --world-size <n>        Total number of workers (default: 1)\n"
              << "  --master-addr <addr>    TCP master address (default: 127.0.0.1)\n"
              << "  --master-port <port>    TCP master port (default: 29500)\n"
              << "  --batch-size <n>        Global batch size for NN training (default: 64)\n"
              << "  --epochs <n>            NN training iterations (default: 1000)\n"
              << "  --seed <n>              Deterministic RNG seed (default: 1337)\n"
              << "  --checkpoint-prefix <p> Checkpoint file prefix (saves <p>.model and <p>.optim)\n"
              << "  --resume-checkpoint     Resume from the checkpoint prefix before training\n";
}

ProgramOptions parse_program_options(int argc, char **argv)
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

        if (arg == "--distributed")
        {
            options.distributed = true;
            continue;
        }
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
        if (arg == "--batch-size")
        {
            options.batch_size = static_cast<std::size_t>(std::stoul(require_value("--batch-size")));
            continue;
        }
        if (arg == "--epochs")
        {
            options.epochs = static_cast<std::size_t>(std::stoul(require_value("--epochs")));
            continue;
        }
        if (arg == "--seed")
        {
            options.seed = static_cast<std::uint64_t>(std::stoull(require_value("--seed")));
            continue;
        }
        if (arg == "--checkpoint-prefix")
        {
            options.checkpoint_prefix = require_value("--checkpoint-prefix");
            continue;
        }
        if (arg == "--resume-checkpoint")
        {
            options.resume_checkpoint = true;
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
    if (options.batch_size == 0)
    {
        throw std::invalid_argument("--batch-size must be positive.");
    }
    if (options.resume_checkpoint && options.checkpoint_prefix.empty())
    {
        throw std::invalid_argument("--resume-checkpoint requires --checkpoint-prefix.");
    }

    return options;
}

Matrix softmax(Matrix &logits)
{
    Matrix probs(logits.numRows(), 1); 

    float max_val = -std::numeric_limits<float>::max();
    for (size_t i = 0; i < probs.numRows(); ++i)
    {
        if (logits(i, 0) > max_val && logits(i, 0) != -1e9)
        {
            max_val = logits(i, 0);
        }
    }

    float sum_exp = 0.0f;
    for (size_t i = 0; i < probs.numRows(); ++i)
    {
        if (logits(i, 0) == -1e9)
        {
            probs(i, 0) = 0.0f; 
        }
        else
        {
            probs(i, 0) = std::exp(logits(i, 0) - max_val);
            sum_exp += probs(i, 0);
        }
    }

    // Normalize the probabilities
    for (size_t i = 0; i < probs.numRows(); ++i)
    {
        probs(i, 0) /= (sum_exp + 1e-10);
    }

    return probs;
}

std::vector<Matrix> get_data_points(const Matrix &data)
{
    std::vector<Matrix> data_points;

    for (size_t i = 0; i < data.numRows(); ++i)
    {
        Matrix p(data.numCols(), 1);
        for (size_t j = 0; j < data.numCols(); ++j)
        {
            p(j, 0) = data(i, j);
        }
        data_points.push_back(p);
    }

    return data_points;
}

Matrix one_hot_encode(int label, int d)
{
    Matrix encoding(d, 1);

    for (size_t i = 0; i < d; ++i)
    {
        encoding(i, 0) = (i == label) ? 1 : 0;
    }

    return encoding;
}

// class labels is row vector
std::pair<Matrix, Matrix> get_random_batch(const Matrix &x_full,
                                           const Matrix &class_labels, // shape: (1, total_samples)
                                           int batch_size,
                                           std::mt19937 &g)
{
    if (batch_size <= 0)
    {
        throw std::invalid_argument("batch_size must be positive.");
    }
    if (class_labels.numRows() != 1)
    {
        throw std::invalid_argument("class_labels must be a row vector (1 x N).");
    }
    if (x_full.numRows() != class_labels.numCols())
    {
        throw std::invalid_argument("x_full rows must match class_labels columns.");
    }

    int total_samples = static_cast<int>(x_full.numRows());
    if (batch_size > total_samples)
    {
        throw std::invalid_argument("batch_size cannot exceed number of samples.");
    }

    std::vector<int> indices(total_samples);
    std::iota(indices.begin(), indices.end(), 0);

    std::shuffle(indices.begin(), indices.end(), g);

    Matrix x_batch(batch_size, x_full.numCols());
    Matrix y_batch(1, batch_size);

    for (int i = 0; i < batch_size; ++i)
    {
        int idx = indices[i];
        for (int j = 0; j < x_full.numCols(); ++j)
        {
            x_batch(i, j) = x_full(idx, j);
        }
        y_batch(0, i) = class_labels(0, idx);
    }

    return {x_batch, y_batch};
}

int main(int argc, char **argv)
{
    const ProgramOptions options = parse_program_options(argc, argv);

    using math::ASTNode;
    using math::Differentiator;
    using math::GD;
    using math::Log;
    using math::Max;
    using math::Min;
    using math::Num;
    using math::OptimParams;
    using math::Var;
    using ml::ActivationLayer;
    using ml::KernelSVM;
    using ml::LinearLayer;
    using ml::LinearRegression;
    using ml::LogisticRegression;
    using ml::RandomFourierSVM;
    using ml::ReLULayer;
    using ml::Sequential;
    using ml::SigmoidLayer;
    using ml::SVM;

    int num_samples = 150;

    const char *iris_env = std::getenv("CTORCH_IRIS_CSV");
    const std::string iris_csv = iris_env ? std::string(iris_env) : std::string("Iris.csv");

    io::CSVReader<5> in(iris_csv);
    in.read_header(io::ignore_extra_column,
                   "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species");

    size_t total_rows = 0;
    float temp1, temp2, temp3, temp4;
    std::string temp5;

    io::CSVReader<5> counter(iris_csv);
    counter.read_header(io::ignore_extra_column,
                        "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species");

    while (counter.read_row(temp1, temp2, temp3, temp4, temp5))
    {
        total_rows++;
    }

    std::mt19937 gen(static_cast<std::mt19937::result_type>(options.seed));
    std::uniform_int_distribution<size_t> dist(0, total_rows - 1);

    std::unordered_set<size_t> selected_indices;
    while (selected_indices.size() < num_samples && selected_indices.size() < total_rows)
    {
        selected_indices.insert(dist(gen));
    }

    float col1, col2, col3, col4; // sepal.length, sepal.width, petal.length, petal.width
    std::string col5;             // variety (Setosa, Versicolor, or Virginica)

    Matrix full_x(num_samples, 4);
    Matrix full_y(1, num_samples);            // row
    Matrix full_y_regression(1, num_samples); // regression
    Matrix full_y_binary(1, num_samples);     // row (we'll convert to binary: is it Setosa or not)

    std::vector<std::string> class_labels;

    size_t row = 0;
    int num_setosa = 0;
    size_t current_row = 0;

    io::CSVReader<5> sampler(iris_csv);
    sampler.read_header(io::ignore_extra_column,
                        "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species");

    while (sampler.read_row(col1, col2, col3, col4, col5))
    {
        // only process rows whose indices are in the selected_indices set
        if (selected_indices.find(current_row) != selected_indices.end())
        {
            class_labels.push_back(col5);

            full_x(row, 0) = col1; // sepal length
            full_x(row, 1) = col2; // sepal width
            full_x(row, 2) = col3; // petal length
            full_x(row, 3) = col4; // petal width

            // For full_y: Setosa = 0, Versicolor = 1, Virginica = 2
            if (col5 == "Iris-setosa")
            {
                full_y(0, row) = 0;
                full_y_binary(0, row) = 1; // Setosa = positive class
                num_setosa++;
            }
            else if (col5 == "Iris-versicolor")
            {
                full_y(0, row) = 1;
                full_y_binary(0, row) = 0; // Not Setosa
            }
            else
            { // Virginica
                full_y(0, row) = 2;
                full_y_binary(0, row) = 0; // Not Setosa
            }

            full_y_regression(0, row) = col3;

            row++;
            if (row == num_samples)
            {
                break;
            }
        }
        current_row++;
    }

    LOG("Number of Setosa samples: " << num_setosa);

    for (size_t j = 0; j < 4; ++j)
    {
        double mean = 0.0, std = 0.0;

        for (size_t i = 0; i < num_samples; ++i)
        {
            mean += full_x(i, j);
        }
        mean /= num_samples;

        for (size_t i = 0; i < num_samples; ++i)
        {
            std += (full_x(i, j) - mean) * (full_x(i, j) - mean);
        }
        std = std::sqrt(std / num_samples);

        for (size_t i = 0; i < num_samples; ++i)
        {
            full_x(i, j) = (full_x(i, j) - mean) / (std + 1e-8);
        }
    }

    size_t train_size = static_cast<size_t>(0.8 * num_samples);
    size_t test_size = num_samples - train_size;

    std::vector<size_t> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    Matrix xTr(train_size, 4), yTr(1, train_size), yTr_binary(1, train_size), yTr_regression(1, train_size);
    Matrix xTe(test_size, 4), yTe(1, test_size), yTe_binary(1, test_size), yTe_regression(1, test_size);

    std::vector<std::string> train_classes;
    std::vector<std::string> test_classes;

    for (size_t i = 0; i < train_size; ++i)
    {
        size_t idx = indices[i];
        for (size_t j = 0; j < 4; ++j)
        {
            xTr(i, j) = full_x(idx, j);
        }
        yTr(0, i) = full_y(0, idx);
        yTr_binary(0, i) = full_y_binary(0, idx);
        yTr_regression(0, i) = full_y_regression(0, idx);
        train_classes.push_back(class_labels[idx]);
    }

    for (size_t i = 0; i < test_size; ++i)
    {
        size_t idx = indices[train_size + i];
        for (size_t j = 0; j < 4; ++j)
        {
            xTe(i, j) = full_x(idx, j);
        }
        yTe(0, i) = full_y(0, idx);
        yTe_binary(0, i) = full_y_binary(0, idx);
        yTe_regression(0, i) = full_y_regression(0, idx);
        test_classes.push_back(class_labels[idx]);
    }

    // Logistic Regression Experiments (binary classification: Setosa vs others)
    Matrix yTr_hyperplane(yTr.numRows(), yTr.numCols());
    Matrix yTe_hyperplane(yTe.numRows(), yTe.numCols());

    for (size_t i = 0; i < yTr.numCols(); ++i)
    {
        yTr_hyperplane(0, i) = yTr_binary(0, i) == 1 ? 1 : -1;
    }

    for (size_t i = 0; i < yTe.numCols(); ++i)
    {
        yTe_hyperplane(0, i) = yTe_binary(0, i) == 1 ? 1 : -1;
    }

    // SVM Experiments (binary classification: Setosa vs others)
    double learning_rate = 0.0001;
    int max_iter = 100;
    double C = 10;
    DataAugmentationType aug = DataAugmentationType::NO_OP;

    SVM model(xTr, yTr_hyperplane, learning_rate, max_iter, C, aug);
    LOG("SVM Scores (Setosa binary classification): " << model.score(xTe, yTe_hyperplane));

    // Perceptron Experiments (binary classification: Setosa vs others)
    ml::Perceptron perceptron_model(xTr, yTr_hyperplane, 100000);
    LOG("Perceptron Scores (Setosa binary classification): " << perceptron_model.score(xTe, yTe_hyperplane));

    LinearRegression regression_model(xTr, yTr_regression, 0.0001, 100);
    LOG("Regression Scores (Setosa binary classification): " << regression_model.score(xTe, yTe_binary, 2.0));

    ml::GaussianNB gnb(xTr, yTr);
    LOG("Gaussian Naive Bayes (3-class Iris): " << gnb.score(xTe, yTe));

    // Create non-linearly separable dataset for KernelSVM and RandomFourierSVM
    LOG("\n=== Creating Non-Linearly Separable Dataset ===");

    // filter out Versicolor and Virginica samples
    std::vector<size_t> nonlinear_indices;
    for (size_t i = 0; i < num_samples; ++i)
    {
        if (class_labels[i] == "Iris-versicolor" || class_labels[i] == "Iris-virginica")
        {
            nonlinear_indices.push_back(i);
        }
    }

    size_t nl_samples = nonlinear_indices.size();
    LOG("Number of samples in non-linear dataset (Versicolor + Virginica): " << nl_samples);

    Matrix nl_X(nl_samples, 2);
    Matrix nl_y(1, nl_samples);
    std::vector<std::string> nl_classes;

    for (size_t i = 0; i < nl_samples; ++i)
    {
        size_t idx = nonlinear_indices[i];

        // Just use petal length and width (normalized values from full_x)
        nl_X(i, 0) = full_x(idx, 2); // petal length
        nl_X(i, 1) = full_x(idx, 3); // petal width

        // Class labels: 1 for Versicolor, -1 for Virginica (SVM format)
        nl_y(0, i) = (class_labels[idx] == "Iris-versicolor") ? 1.0 : -1.0;
        nl_classes.push_back(class_labels[idx]);
    }

    size_t nl_train_size = static_cast<size_t>(0.8 * nl_samples);
    size_t nl_test_size = nl_samples - nl_train_size;

    std::vector<size_t> nl_indices(nl_samples);
    std::iota(nl_indices.begin(), nl_indices.end(), 0);
    std::shuffle(nl_indices.begin(), nl_indices.end(), gen);

    Matrix nl_xTr(nl_train_size, 2), nl_yTr(1, nl_train_size);
    Matrix nl_xTe(nl_test_size, 2), nl_yTe(1, nl_test_size);
    std::vector<std::string> nl_train_classes, nl_test_classes;

    for (size_t i = 0; i < nl_train_size; ++i)
    {
        size_t idx = nl_indices[i];
        for (size_t j = 0; j < 2; ++j)
        {
            nl_xTr(i, j) = nl_X(idx, j);
        }
        nl_yTr(0, i) = nl_y(0, idx);
        nl_train_classes.push_back(nl_classes[idx]);
    }

    for (size_t i = 0; i < nl_test_size; ++i)
    {
        size_t idx = nl_indices[nl_train_size + i];
        for (size_t j = 0; j < 2; ++j)
        {
            nl_xTe(i, j) = nl_X(idx, j);
        }
        nl_yTe(0, i) = nl_y(0, idx);
        nl_test_classes.push_back(nl_classes[idx]);
    }

    LOG("Non-linear dataset training size: " << nl_train_size);
    LOG("Non-linear dataset testing size: " << nl_test_size);

    std::cout << "\nSample data from non-linear training set:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), nl_train_size); ++i)
    {
        std::cout << "Features: ["
                  << nl_xTr(i, 0) << ", " << nl_xTr(i, 1) << "], "
                  << "Class: " << nl_train_classes[i] << std::endl;
    }

    // Linear SVM
    SVM linear_svm(nl_xTr, nl_yTr, learning_rate, max_iter, C, aug);
    LOG("\nLinear SVM Score on non-linear dataset: " << linear_svm.score(nl_xTe, nl_yTe));

    // k-NN
    ml::KNN knn_model(5, nl_xTr, nl_yTr);
    LOG("\nk-NN score on non-linear dataset: " << knn_model.score(nl_xTe, nl_yTe));

    // Neural Network
    ml::Sequential nn_model;

    nn_model.add_layer(std::make_shared<ml::LinearLayer>(4, 32));
    nn_model.add_layer(std::make_shared<ml::ReLULayer>());
    nn_model.add_layer(std::make_shared<ml::LinearLayer>(32, 32));
    nn_model.add_layer(std::make_shared<ml::ReLULayer>());
    nn_model.add_layer(std::make_shared<ml::LinearLayer>(32, 3));
    if (options.distributed)
    {
        if (options.batch_size % static_cast<std::size_t>(options.world_size) != 0)
        {
            throw std::invalid_argument("--batch-size must be divisible by --world-size in distributed mode.");
        }

        const std::size_t local_batch_size = options.batch_size / static_cast<std::size_t>(options.world_size);
        auto [local_xTr, local_yTr] = ctorch::distributed::equal_row_shard(xTr, yTr, options.rank, options.world_size);
        if (local_batch_size == 0)
        {
            throw std::invalid_argument("Distributed local batch size must be positive.");
        }
        if (local_batch_size > local_xTr.numRows())
        {
            throw std::invalid_argument("Distributed local batch size exceeds the rank's data shard.");
        }

        std::mt19937 batch_gen(static_cast<std::mt19937::result_type>(options.seed + static_cast<std::uint64_t>(options.rank)));
        ctorch::distributed::TcpProcessGroup group(options.master_address, options.master_port, options.rank, options.world_size);

        ml::NN_SGD optimizer(nn_model.parameters(), 0.01f, local_batch_size);
        if (options.resume_checkpoint)
        {
            ctorch::distributed::load_distributed_checkpoint(group, options.checkpoint_prefix, nn_model, optimizer);
        }
        else
        {
            ctorch::distributed::synchronize_sequential_model(group, nn_model);
        }

        for (std::size_t i = 0; i < options.epochs; ++i)
        {
            auto [data, labels] = get_random_batch(local_xTr, local_yTr, static_cast<int>(local_batch_size), batch_gen);
            const std::vector<Matrix> data_points = get_data_points(data);
            ctorch::distributed::distributed_parallel_backpropagate_batch(
                group,
                nn_model,
                optimizer,
                data_points,
                [&](ml::Sequential &local_model, const Matrix &p, std::size_t labels_idx)
                {
                    Matrix logits = local_model.forward(p);
                    local_model.backward(softmax(logits) - one_hot_encode(labels(0, labels_idx), 3));
                });
        }

        if (!options.checkpoint_prefix.empty())
        {
            ctorch::distributed::save_distributed_checkpoint(group, options.checkpoint_prefix, nn_model, optimizer);
        }
    }
    else
    {
        std::mt19937 batch_gen(static_cast<std::mt19937::result_type>(options.seed));
        ml::NN_SGD optimizer(nn_model.parameters(), 0.01f, options.batch_size);

        if (options.resume_checkpoint)
        {
            if (!nn_model.load(options.checkpoint_prefix + ".model"))
            {
                throw std::runtime_error("Failed to load classification checkpoint model.");
            }
            if (!optimizer.load_state(options.checkpoint_prefix + ".optim"))
            {
                throw std::runtime_error("Failed to load classification checkpoint optimizer.");
            }
        }

        for (std::size_t i = 0; i < options.epochs; ++i)
        {
            optimizer.zero_grad();
            auto [data, labels] = get_random_batch(xTr, yTr, static_cast<int>(options.batch_size), batch_gen);
            const std::vector<Matrix> data_points = get_data_points(data);
            ml::parallel_backpropagate_batch(
                nn_model,
                data_points,
                [&](ml::Sequential &local_model, const Matrix &p, std::size_t labels_idx)
                {
                    Matrix logits = local_model.forward(p);
                    local_model.backward(softmax(logits) - one_hot_encode(labels(0, labels_idx), 3));
            });
            optimizer.step();
        }

        if (!options.checkpoint_prefix.empty())
        {
            if (!nn_model.save(options.checkpoint_prefix + ".model"))
            {
                throw std::runtime_error("Failed to save classification checkpoint model.");
            }
            if (!optimizer.save_state(options.checkpoint_prefix + ".optim"))
            {
                throw std::runtime_error("Failed to save classification checkpoint optimizer.");
            }
        }
    }

    float num_correct = 0.0f;

    size_t labels_idx = 0;
    for (const Matrix &p : get_data_points(xTe))
    {
        Matrix logits = nn_model.forward(p);
        int max_index = 0;
        float max_value = logits(0, 0);

        for (int i = 1; i < logits.numRows(); ++i)
        {
            if (logits(i, 0) > max_value)
            {
                max_value = logits(i, 0);
                max_index = i;
            }
        }

        if (max_index == yTe(0, labels_idx))
        {
            num_correct += 1.0;
        }

        labels_idx++;
    }

    LOG(num_correct << " " << xTe.numRows());

    LOG("NN predicting flower: " << static_cast<double>((num_correct / xTe.numRows())));
}
