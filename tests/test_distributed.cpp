#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <cmath>
#include <filesystem>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "distributed/distributed_sync.hpp"
#include "distributed/distributed_checkpoint.hpp"
#include "distributed/distributed_optimizer.hpp"
#include "distributed/distributed_training.hpp"
#include "distributed/tcp_process_group.hpp"
#include "math/optim.hpp"
#include "math/matrix.hpp"
#include "ml/nn.hpp"
#include "ml/nn_optim.hpp"

namespace
{
struct WorkerResult
{
    std::vector<Matrix> parameters;
    Matrix reduced;
};

struct ThetaResult
{
    std::unordered_map<std::string, double> theta;
};

std::uint16_t reserve_port()
{
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
    {
        throw std::runtime_error("socket: unable to reserve a test port.");
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(0);

    if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0)
    {
        ::close(fd);
        throw std::runtime_error("bind: unable to reserve a test port.");
    }

    socklen_t len = sizeof(addr);
    if (::getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &len) != 0)
    {
        ::close(fd);
        throw std::runtime_error("getsockname: unable to inspect the reserved port.");
    }

    const std::uint16_t port = ntohs(addr.sin_port);
    ::close(fd);
    return port;
}

ml::Sequential make_reference_model()
{
    ml::Sequential model;
    model.add_layer(std::make_shared<ml::LinearLayer>(2, 3));
    model.add_layer(std::make_shared<ml::ReLULayer>());
    model.add_layer(std::make_shared<ml::LinearLayer>(3, 1));

    auto params = model.parameters();
    if (params.size() != 4)
    {
        throw std::runtime_error("Reference model parameter layout changed unexpectedly.");
    }

    *params[0].first = Matrix({{0.2, -0.1}, {0.4, 0.3}, {-0.2, 0.5}});
    *params[1].first = Matrix({{0.1}, {-0.2}, {0.05}});
    *params[2].first = Matrix({{0.6, -0.4, 0.2}});
    *params[3].first = Matrix({{0.15}});

    return model;
}

ml::Sequential make_peer_model()
{
    ml::Sequential model;
    model.add_layer(std::make_shared<ml::LinearLayer>(2, 3));
    model.add_layer(std::make_shared<ml::ReLULayer>());
    model.add_layer(std::make_shared<ml::LinearLayer>(3, 1));
    return model;
}

ml::Sequential make_mismatched_model()
{
    ml::Sequential model;
    model.add_layer(std::make_shared<ml::LinearLayer>(2, 4));
    model.add_layer(std::make_shared<ml::ReLULayer>());
    model.add_layer(std::make_shared<ml::LinearLayer>(4, 1));
    return model;
}

std::vector<Matrix> snapshot_parameters(const ml::Sequential &model)
{
    std::vector<Matrix> snapshot;
    const std::size_t parameter_count = model.parameter_count();
    snapshot.reserve(parameter_count);

    for (std::size_t index = 0; index < parameter_count; ++index)
    {
        Matrix parameter;
        if (!model.get_parameter(index, parameter))
        {
            throw std::runtime_error("Unable to snapshot Sequential parameters.");
        }
        snapshot.push_back(parameter);
    }

    return snapshot;
}

void assign_gradients(ml::Sequential &model, const std::vector<Matrix> &gradients)
{
    auto params = model.parameters();
    if (params.size() != gradients.size())
    {
        throw std::runtime_error("Gradient layout mismatch while assigning test gradients.");
    }

    for (std::size_t index = 0; index < params.size(); ++index)
    {
        if (!params[index].second)
        {
            throw std::runtime_error("Encountered a null gradient buffer while assigning test gradients.");
        }
        if (params[index].second->numRows() != gradients[index].numRows() ||
            params[index].second->numCols() != gradients[index].numCols())
        {
            throw std::runtime_error("Gradient shape mismatch while assigning test gradients.");
        }
        *params[index].second = gradients[index];
    }
}

bool matrices_match(const Matrix &lhs, const Matrix &rhs)
{
    if (lhs.numRows() != rhs.numRows() || lhs.numCols() != rhs.numCols())
    {
        return false;
    }

    for (std::size_t row = 0; row < lhs.numRows(); ++row)
    {
        for (std::size_t col = 0; col < lhs.numCols(); ++col)
        {
            if (std::abs(lhs(row, col) - rhs(row, col)) > 1e-9)
            {
                return false;
            }
        }
    }

    return true;
}

class LogisticLossFunction final : public LossFunction
{
public:
    std::shared_ptr<math::ASTNode> sample_loss(const Matrix &x, double y) const override
    {
        std::shared_ptr<math::ASTNode> logits = math::Num(0.0);
        for (std::size_t i = 0; i < x.numCols(); ++i)
        {
            logits = logits + math::Var("w" + std::to_string(i)) * math::Num(x.at(0, i));
        }

        logits = logits + math::Var("b");
        std::shared_ptr<math::ASTNode> y_hat = math::Sigmoid(logits);
        return -math::Num(y) * math::Log(y_hat) -
               (math::Num(1.0) - math::Num(y)) * math::Log(math::Num(1.0) - y_hat);
    }

    std::shared_ptr<math::ASTNode> regularizer() const override
    {
        return math::Num(0.0);
    }
};

std::pair<Matrix, Matrix> make_logistic_dataset()
{
    constexpr std::size_t grid_size = 8;
    constexpr std::size_t sample_count = grid_size * grid_size;
    Matrix x(sample_count, 2);
    Matrix y(1, sample_count);

    std::size_t index = 0;
    for (std::size_t row = 0; row < grid_size; ++row)
    {
        const double x0 = -1.5 + 3.0 * static_cast<double>(row) / static_cast<double>(grid_size - 1);
        for (std::size_t col = 0; col < grid_size; ++col)
        {
            const double x1 = -1.5 + 3.0 * static_cast<double>(col) / static_cast<double>(grid_size - 1);
            x(index, 0) = x0;
            x(index, 1) = x1;
            y(0, index) = (x0 + 0.75 * x1 > 0.1) ? 1.0 : 0.0;
            ++index;
        }
    }

    return {x, y};
}

bool theta_matches(
    const std::unordered_map<std::string, double> &lhs,
    const std::unordered_map<std::string, double> &rhs,
    double tolerance = 1e-8)
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }

    for (const auto &entry : lhs)
    {
        const auto it = rhs.find(entry.first);
        if (it == rhs.end())
        {
            return false;
        }
        if (std::abs(entry.second - it->second) > tolerance)
        {
            return false;
        }
    }

    return true;
}

template <typename Fn>
std::unique_ptr<ctorch::distributed::TcpProcessGroup> connect_with_retry(
    const std::string &address,
    std::uint16_t port,
    int rank,
    int world_size,
    Fn &&sleep_fn)
{
    for (int attempt = 0; attempt < 50; ++attempt)
    {
        try
        {
            return std::make_unique<ctorch::distributed::TcpProcessGroup>(address, port, rank, world_size);
        }
        catch (const std::runtime_error &)
        {
            if (attempt == 49)
            {
                throw;
            }
            sleep_fn();
        }
    }

    throw std::runtime_error("Unreachable connect_with_retry path.");
}
} // namespace

TEST(Distributed, SequentialArchitectureSignatureIsStable)
{
    const ml::Sequential reference = make_reference_model();
    const ml::Sequential same_shape = make_reference_model();
    const ml::Sequential mismatched = make_mismatched_model();

    EXPECT_EQ(
        reference.architecture_signature(),
        "layers=3;params=4;Linear(2,3)|ReLU|Linear(3,1)");
    EXPECT_EQ(reference.architecture_signature(), same_shape.architecture_signature());
    EXPECT_NE(reference.architecture_signature(), mismatched.architecture_signature());
}

TEST(Distributed, EqualRowShardSplitsMatricesEvenly)
{
    Matrix x(6, 2);
    Matrix y(1, 6);

    for (std::size_t row = 0; row < x.numRows(); ++row)
    {
        x(row, 0) = static_cast<double>(row);
        x(row, 1) = static_cast<double>(row + 10);
        y(0, row) = static_cast<double>(row + 100);
    }

    const auto [x_shard, y_shard] = ctorch::distributed::equal_row_shard(x, y, 1, 3);

    ASSERT_EQ(x_shard.numRows(), 2u);
    ASSERT_EQ(x_shard.numCols(), 2u);
    ASSERT_EQ(y_shard.numRows(), 1u);
    ASSERT_EQ(y_shard.numCols(), 2u);
    EXPECT_DOUBLE_EQ(x_shard(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(x_shard(0, 1), 12.0);
    EXPECT_DOUBLE_EQ(x_shard(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(x_shard(1, 1), 13.0);
    EXPECT_DOUBLE_EQ(y_shard(0, 0), 102.0);
    EXPECT_DOUBLE_EQ(y_shard(0, 1), 103.0);
}

TEST(Distributed, TcpProcessGroupSynchronizesAndAllReduces)
{
    const std::uint16_t port = reserve_port();
    std::promise<WorkerResult> root_promise;
    std::promise<WorkerResult> peer_promise;
    std::future<WorkerResult> root_future = root_promise.get_future();
    std::future<WorkerResult> peer_future = peer_promise.get_future();

    std::thread root_thread([&]() {
        try
        {
            ml::Sequential model = make_reference_model();
            ctorch::distributed::TcpProcessGroup group("127.0.0.1", port, 0, 2);
            ctorch::distributed::synchronize_sequential_model(group, model);

            WorkerResult result;
            result.parameters = snapshot_parameters(model);

            Matrix value({{1.0, 2.0}});
            group.allreduce_sum(value);
            result.reduced = value;

            root_promise.set_value(std::move(result));
        }
        catch (...)
        {
            root_promise.set_exception(std::current_exception());
        }
    });

    std::thread peer_thread([&]() {
        try
        {
            auto sleep_fn = []() {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            };
            ml::Sequential model = make_peer_model();
            std::unique_ptr<ctorch::distributed::TcpProcessGroup> group =
                connect_with_retry("127.0.0.1", port, 1, 2, sleep_fn);
            ctorch::distributed::synchronize_sequential_model(*group, model);

            WorkerResult result;
            result.parameters = snapshot_parameters(model);

            Matrix value({{3.0, 4.0}});
            group->allreduce_sum(value);
            result.reduced = value;

            peer_promise.set_value(std::move(result));
        }
        catch (...)
        {
            peer_promise.set_exception(std::current_exception());
        }
    });

    root_thread.join();
    peer_thread.join();

    const WorkerResult root_result = root_future.get();
    const WorkerResult peer_result = peer_future.get();

    const std::vector<Matrix> expected = {
        Matrix({{0.2, -0.1}, {0.4, 0.3}, {-0.2, 0.5}}),
        Matrix({{0.1}, {-0.2}, {0.05}}),
        Matrix({{0.6, -0.4, 0.2}}),
        Matrix({{0.15}})
    };

    ASSERT_EQ(root_result.parameters.size(), expected.size());
    ASSERT_EQ(peer_result.parameters.size(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        EXPECT_TRUE(matrices_match(root_result.parameters[index], expected[index]));
        EXPECT_TRUE(matrices_match(peer_result.parameters[index], expected[index]));
    }

    EXPECT_TRUE(matrices_match(root_result.reduced, Matrix({{4.0, 6.0}})));
    EXPECT_TRUE(matrices_match(peer_result.reduced, Matrix({{4.0, 6.0}})));
}

TEST(Distributed, DistributedParallelBackpropMatchesSerialUpdate)
{
    const std::uint16_t port = reserve_port();
    const std::vector<Matrix> samples = {
        Matrix({{1.0}, {0.0}}),
        Matrix({{0.5}, {1.0}}),
        Matrix({{-1.0}, {0.25}}),
        Matrix({{0.0}, {0.75}})
    };

    const std::vector<Matrix> upstreams = {
        Matrix({{1.0}}),
        Matrix({{-0.5}}),
        Matrix({{0.25}}),
        Matrix({{2.0}})
    };

    ml::Sequential serial = make_reference_model();
    ml::NN_SGD serial_optimizer(serial.parameters(), 0.1f, samples.size());
    serial_optimizer.zero_grad();
    for (std::size_t index = 0; index < samples.size(); ++index)
    {
        serial.forward(samples[index]);
        serial.backward(upstreams[index]);
    }
    serial_optimizer.step();
    const std::vector<Matrix> serial_snapshot = snapshot_parameters(serial);

    std::promise<WorkerResult> root_promise;
    std::promise<WorkerResult> peer_promise;
    std::future<WorkerResult> root_future = root_promise.get_future();
    std::future<WorkerResult> peer_future = peer_promise.get_future();

    std::thread root_thread([&]() {
        try
        {
            ml::Sequential model = make_reference_model();
            ctorch::distributed::TcpProcessGroup group("127.0.0.1", port, 0, 2);
            ctorch::distributed::synchronize_sequential_model(group, model);

            std::vector<Matrix> local_samples(samples.begin(), samples.begin() + 2);
            std::vector<Matrix> local_upstreams(upstreams.begin(), upstreams.begin() + 2);
            ml::NN_SGD optimizer(model.parameters(), 0.1f, local_samples.size());
            ctorch::distributed::distributed_parallel_backpropagate_batch(
                group,
                model,
                optimizer,
                local_samples,
                [&](ml::Sequential &local_model, const Matrix &sample, std::size_t sample_index)
                {
                    Matrix logits = local_model.forward(sample);
                    local_model.backward(local_upstreams[sample_index]);
                    (void)logits;
                });

            WorkerResult result;
            result.parameters = snapshot_parameters(model);
            root_promise.set_value(std::move(result));
        }
        catch (...)
        {
            root_promise.set_exception(std::current_exception());
        }
    });

    std::thread peer_thread([&]() {
        try
        {
            auto sleep_fn = []() {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            };
            ml::Sequential model = make_peer_model();
            std::unique_ptr<ctorch::distributed::TcpProcessGroup> group =
                connect_with_retry("127.0.0.1", port, 1, 2, sleep_fn);
            ctorch::distributed::synchronize_sequential_model(*group, model);

            std::vector<Matrix> local_samples(samples.begin() + 2, samples.end());
            std::vector<Matrix> local_upstreams(upstreams.begin() + 2, upstreams.end());
            ml::NN_SGD optimizer(model.parameters(), 0.1f, local_samples.size());
            ctorch::distributed::distributed_parallel_backpropagate_batch(
                *group,
                model,
                optimizer,
                local_samples,
                [&](ml::Sequential &local_model, const Matrix &sample, std::size_t sample_index)
                {
                    Matrix logits = local_model.forward(sample);
                    local_model.backward(local_upstreams[sample_index]);
                    (void)logits;
                });

            WorkerResult result;
            result.parameters = snapshot_parameters(model);
            peer_promise.set_value(std::move(result));
        }
        catch (...)
        {
            peer_promise.set_exception(std::current_exception());
        }
    });

    root_thread.join();
    peer_thread.join();

    const WorkerResult root_result = root_future.get();
    const WorkerResult peer_result = peer_future.get();

    ASSERT_EQ(root_result.parameters.size(), serial_snapshot.size());
    ASSERT_EQ(peer_result.parameters.size(), serial_snapshot.size());
    for (std::size_t index = 0; index < serial_snapshot.size(); ++index)
    {
        EXPECT_TRUE(matrices_match(root_result.parameters[index], serial_snapshot[index]));
        EXPECT_TRUE(matrices_match(peer_result.parameters[index], serial_snapshot[index]));
    }
}

TEST(Distributed, OptimizerCheckpointRoundTripPreservesFutureSteps)
{
    const std::string prefix = "distributed_checkpoint_roundtrip_test";

    std::vector<Matrix> first_gradients = {
        Matrix({{1.0, -0.5}, {0.25, 0.75}, {-0.25, 0.5}}),
        Matrix({{0.1}, {-0.2}, {0.05}}),
        Matrix({{0.6, -0.4, 0.2}}),
        Matrix({{0.15}})
    };
    std::vector<Matrix> second_gradients = {
        Matrix({{0.5, 0.25}, {-0.75, 0.1}, {0.0, -0.3}}),
        Matrix({{0.2}, {0.4}, {-0.1}}),
        Matrix({{-0.3, 0.7, -0.2}}),
        Matrix({{-0.05}})
    };
    std::vector<Matrix> third_gradients = {
        Matrix({{-0.4, 0.3}, {0.6, -0.2}, {0.15, 0.45}}),
        Matrix({{0.05}, {0.1}, {-0.2}}),
        Matrix({{0.25, -0.15, 0.35}}),
        Matrix({{0.08}})
    };

    ml::Sequential saved = make_reference_model();
    ml::Sequential restored = make_reference_model();
    ml::NN_SGD saved_optimizer(saved.parameters(), 0.1f, 4);
    ml::NN_SGD restored_optimizer(restored.parameters(), 0.1f, 4);

    assign_gradients(saved, first_gradients);
    saved_optimizer.step();
    assign_gradients(saved, second_gradients);
    saved_optimizer.step();

    ASSERT_TRUE(saved.save(prefix + ".model"));
    ASSERT_TRUE(saved_optimizer.save_state(prefix + ".optim"));

    ASSERT_TRUE(restored.load(prefix + ".model"));
    ASSERT_TRUE(restored_optimizer.load_state(prefix + ".optim"));

    assign_gradients(saved, third_gradients);
    assign_gradients(restored, third_gradients);
    saved_optimizer.step();
    restored_optimizer.step();

    const std::vector<Matrix> saved_snapshot = snapshot_parameters(saved);
    const std::vector<Matrix> restored_snapshot = snapshot_parameters(restored);
    ASSERT_EQ(saved_snapshot.size(), restored_snapshot.size());
    for (std::size_t index = 0; index < saved_snapshot.size(); ++index)
    {
        EXPECT_TRUE(matrices_match(saved_snapshot[index], restored_snapshot[index]));
    }

    std::filesystem::remove(prefix + ".model");
    std::filesystem::remove(prefix + ".optim");
}

TEST(Distributed, DistributedAstOptimizerMatchesSerialLogisticRegression)
{
    const std::uint16_t port = reserve_port();
    const auto [x, y] = make_logistic_dataset();
    const auto loss_function = std::make_shared<LogisticLossFunction>();
    const math::OptimParams optim_params(
        math::OptimType::GD,
        0.15,
        200,
        Matrix(),
        Matrix(),
        1);
    const std::unordered_map<std::string, double> initial_theta = {
        {"w0", 0.0},
        {"w1", 0.0},
        {"b", 0.0},
    };

    math::Optimizer serial_optimizer(optim_params);
    const std::unordered_map<std::string, double> serial_theta =
        serial_optimizer.optimize(loss_function, x, y, initial_theta);

    std::promise<ThetaResult> root_promise;
    std::promise<ThetaResult> peer_promise;
    std::future<ThetaResult> root_future = root_promise.get_future();
    std::future<ThetaResult> peer_future = peer_promise.get_future();

    std::thread root_thread([&]() {
        try
        {
            ctorch::distributed::TcpProcessGroup group("127.0.0.1", port, 0, 2);
            ctorch::distributed::DistributedOptimizer optimizer(group, optim_params);
            ThetaResult result;
            result.theta = optimizer.optimize(loss_function, x, y, initial_theta);
            root_promise.set_value(std::move(result));
        }
        catch (...)
        {
            root_promise.set_exception(std::current_exception());
        }
    });

    std::thread peer_thread([&]() {
        try
        {
            auto sleep_fn = []() {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            };
            std::unique_ptr<ctorch::distributed::TcpProcessGroup> group =
                connect_with_retry("127.0.0.1", port, 1, 2, sleep_fn);
            ctorch::distributed::DistributedOptimizer optimizer(*group, optim_params);
            ThetaResult result;
            result.theta = optimizer.optimize(loss_function, x, y, initial_theta);
            peer_promise.set_value(std::move(result));
        }
        catch (...)
        {
            peer_promise.set_exception(std::current_exception());
        }
    });

    root_thread.join();
    peer_thread.join();

    const ThetaResult root_result = root_future.get();
    const ThetaResult peer_result = peer_future.get();

    ASSERT_TRUE(theta_matches(root_result.theta, serial_theta));
    ASSERT_TRUE(theta_matches(peer_result.theta, serial_theta));
    ASSERT_TRUE(theta_matches(root_result.theta, peer_result.theta));
}

TEST(Distributed, SynchronizeSequentialModelRejectsMismatchedArchitecture)
{
    const std::uint16_t port = reserve_port();
    std::promise<void> root_promise;
    std::promise<void> peer_promise;
    std::future<void> root_future = root_promise.get_future();
    std::future<void> peer_future = peer_promise.get_future();

    std::thread root_thread([&]() {
        try
        {
            ml::Sequential model = make_reference_model();
            ctorch::distributed::TcpProcessGroup group("127.0.0.1", port, 0, 2);
            ctorch::distributed::synchronize_sequential_model(group, model);
            root_promise.set_value();
        }
        catch (...)
        {
            root_promise.set_exception(std::current_exception());
        }
    });

    std::thread peer_thread([&]() {
        try
        {
            auto sleep_fn = []() {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            };
            ml::Sequential model = make_mismatched_model();
            std::unique_ptr<ctorch::distributed::TcpProcessGroup> group =
                connect_with_retry("127.0.0.1", port, 1, 2, sleep_fn);
            ctorch::distributed::synchronize_sequential_model(*group, model);
            peer_promise.set_value();
        }
        catch (...)
        {
            peer_promise.set_exception(std::current_exception());
        }
    });

    root_thread.join();
    peer_thread.join();

    EXPECT_THROW(root_future.get(), std::runtime_error);
    EXPECT_THROW(peer_future.get(), std::runtime_error);
}
