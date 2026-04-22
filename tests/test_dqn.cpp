#include <gtest/gtest.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>
#include "dqn.hpp"
#include "math/matrix.hpp"
#include "ml/parallel_training.hpp"
#include "ml/nn.hpp"

TEST(DQN, GreedyActOnlyUsesValidActions) {
    for (int trial = 0; trial < 150; ++trial) {
        ml::Sequential q;
        q.add_layer(std::make_shared<ml::LinearLayer>(4, 8));
        q.add_layer(std::make_shared<ml::ReLULayer>());
        q.add_layer(std::make_shared<ml::LinearLayer>(8, 6));

        ml::Sequential t;
        t.add_layer(std::make_shared<ml::LinearLayer>(4, 8));
        t.add_layer(std::make_shared<ml::ReLULayer>());
        t.add_layer(std::make_shared<ml::LinearLayer>(8, 6));

        DQNAgent agent(q, t, 0.0f, 0.0f, 0.9f, 0.9f, 0.001f, 32, 500, 5);
        agent.set_epsilon(0.0f);

        std::vector<float> state = {0.2f, -0.1f, 0.4f, 0.0f};
        std::vector<int> valid = {0, 2, 5};
        int a = agent.act(state, valid);
        EXPECT_NE(std::find(valid.begin(), valid.end(), a), valid.end())
            << "trial " << trial << " returned invalid action " << a;
    }
}

TEST(DQN, SingleValidAction) {
    ml::Sequential q;
    q.add_layer(std::make_shared<ml::LinearLayer>(2, 4));
    q.add_layer(std::make_shared<ml::ReLULayer>());
    q.add_layer(std::make_shared<ml::LinearLayer>(4, 3));

    ml::Sequential t;
    t.add_layer(std::make_shared<ml::LinearLayer>(2, 4));
    t.add_layer(std::make_shared<ml::ReLULayer>());
    t.add_layer(std::make_shared<ml::LinearLayer>(4, 3));

    DQNAgent agent(q, t, 0.0f, 0.0f, 0.9f, 0.9f, 0.001f, 8, 200, 1);
    agent.set_epsilon(0.0f);

    std::vector<float> state = {1.0f, -1.0f};
    std::vector<int> valid = {2};
    EXPECT_EQ(agent.act(state, valid), 2);
}

TEST(Sequential, CopyParametersFromSyncsWeights) {
    ml::Sequential a;
    a.add_layer(std::make_shared<ml::LinearLayer>(2, 3));
    a.add_layer(std::make_shared<ml::ReLULayer>());
    a.add_layer(std::make_shared<ml::LinearLayer>(3, 2));

    ml::Sequential b;
    b.add_layer(std::make_shared<ml::LinearLayer>(2, 3));
    b.add_layer(std::make_shared<ml::ReLULayer>());
    b.add_layer(std::make_shared<ml::LinearLayer>(3, 2));

    auto pa = a.parameters();
    ASSERT_GE(pa.size(), 2u);
    *pa[0].first = Matrix({{1.0, 0.0}, {0.0, 1.0}, {2.0, -1.0}});
    *pa[1].first = Matrix({{0.5}, {0.0}, {-0.5}});

    ASSERT_TRUE(b.copy_parameters_from(a));

    Matrix x({{1.0}, {2.0}});
    Matrix ya = a.forward(x);
    Matrix yb = b.forward(x);
    for (size_t i = 0; i < ya.numRows(); ++i) {
        EXPECT_NEAR(ya(i, 0), yb(i, 0), 1e-9);
    }
}

TEST(Sequential, LoadRejectsTruncatedFile) {
    ml::Sequential model;
    model.add_layer(std::make_shared<ml::LinearLayer>(2, 3));
    model.add_layer(std::make_shared<ml::ReLULayer>());
    model.add_layer(std::make_shared<ml::LinearLayer>(3, 1));

    const std::string path = "sequential_truncated_test.bin";
    ASSERT_TRUE(model.save(path));

    std::ifstream in(path, std::ios::binary);
    ASSERT_TRUE(in.is_open());
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    ASSERT_GT(bytes.size(), 1u);

    bytes.pop_back();
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();

    EXPECT_FALSE(model.load(path));
    std::filesystem::remove(path);
}

TEST(Sequential, ParallelBackpropBatchMatchesSerialGradients) {
    ml::Sequential seed;
    seed.add_layer(std::make_shared<ml::LinearLayer>(2, 3));
    seed.add_layer(std::make_shared<ml::ReLULayer>());
    seed.add_layer(std::make_shared<ml::LinearLayer>(3, 1));

    auto seed_params = seed.parameters();
    ASSERT_EQ(seed_params.size(), 4u);
    *seed_params[0].first = Matrix({{0.2, -0.1}, {0.4, 0.3}, {-0.2, 0.5}});
    *seed_params[1].first = Matrix({{0.1}, {-0.2}, {0.05}});
    *seed_params[2].first = Matrix({{0.6, -0.4, 0.2}});
    *seed_params[3].first = Matrix({{0.15}});

    ml::Sequential serial;
    serial.add_layer(std::make_shared<ml::LinearLayer>(2, 3));
    serial.add_layer(std::make_shared<ml::ReLULayer>());
    serial.add_layer(std::make_shared<ml::LinearLayer>(3, 1));
    ASSERT_TRUE(serial.copy_parameters_from(seed));

    ml::Sequential parallel;
    parallel.add_layer(std::make_shared<ml::LinearLayer>(2, 3));
    parallel.add_layer(std::make_shared<ml::ReLULayer>());
    parallel.add_layer(std::make_shared<ml::LinearLayer>(3, 1));
    ASSERT_TRUE(parallel.copy_parameters_from(seed));

    std::vector<Matrix> samples = {
        Matrix({{1.0}, {0.0}}),
        Matrix({{0.5}, {1.0}}),
        Matrix({{-1.0}, {0.25}}),
        Matrix({{0.0}, {0.75}})
    };

    std::vector<Matrix> upstreams = {
        Matrix({{1.0}}),
        Matrix({{-0.5}}),
        Matrix({{0.25}}),
        Matrix({{2.0}})
    };

    ml::NN_SGD serial_optimizer(serial.parameters(), 0.1f, samples.size());
    serial_optimizer.zero_grad();
    for (size_t i = 0; i < samples.size(); ++i) {
        serial.forward(samples[i]);
        serial.backward(upstreams[i]);
    }

    ml::NN_SGD parallel_optimizer(parallel.parameters(), 0.1f, samples.size());
    parallel_optimizer.zero_grad();
    ml::parallel_backpropagate_batch(
        parallel,
        samples,
        [&](ml::Sequential& local_model, const Matrix& sample, std::size_t sample_index) {
            local_model.forward(sample);
            local_model.backward(upstreams[sample_index]);
        });

    auto serial_params = serial.parameters();
    auto parallel_params = parallel.parameters();
    ASSERT_EQ(serial_params.size(), parallel_params.size());
    for (size_t i = 0; i < serial_params.size(); ++i) {
        ASSERT_EQ(serial_params[i].second->numRows(), parallel_params[i].second->numRows());
        ASSERT_EQ(serial_params[i].second->numCols(), parallel_params[i].second->numCols());
        for (size_t row = 0; row < serial_params[i].second->numRows(); ++row) {
            for (size_t col = 0; col < serial_params[i].second->numCols(); ++col) {
                EXPECT_NEAR(
                    (*serial_params[i].second)(row, col),
                    (*parallel_params[i].second)(row, col),
                    1e-9);
            }
        }
    }
}

TEST(DQN, InvalidUpdateFrequencyThrows) {
    ml::Sequential q;
    q.add_layer(std::make_shared<ml::LinearLayer>(2, 4));
    q.add_layer(std::make_shared<ml::ReLULayer>());
    q.add_layer(std::make_shared<ml::LinearLayer>(4, 3));

    ml::Sequential t;
    t.add_layer(std::make_shared<ml::LinearLayer>(2, 4));
    t.add_layer(std::make_shared<ml::ReLULayer>());
    t.add_layer(std::make_shared<ml::LinearLayer>(4, 3));

    EXPECT_THROW(
        DQNAgent(q, t, 0.1f, 0.01f, 0.99f, 0.9f, 0.001f, 8, 200, 0),
        std::invalid_argument);
}
