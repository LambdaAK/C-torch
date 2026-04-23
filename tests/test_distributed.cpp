#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <cmath>
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
#include "distributed/tcp_process_group.hpp"
#include "math/matrix.hpp"
#include "ml/nn.hpp"

namespace
{
struct WorkerResult
{
    std::vector<Matrix> parameters;
    Matrix reduced;
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
