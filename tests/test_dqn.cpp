#include <gtest/gtest.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>
#include "dqn.hpp"
#include "math/matrix.hpp"
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
