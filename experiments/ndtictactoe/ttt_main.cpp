/*
  Commands to compile this file with different optimization levels:

  1. g++ -std=c++20 -I../../lib ttt_main.cpp ./tictactoe.cpp ./replaymemory.cpp ./dqn.cpp ./sample.cpp ../../lib/math/matrix.cpp ../../lib/ml/nn.cpp -o ttt_main
  // no flags: no optimizations

  2. g++ -std=c++20 -O1 -I../../lib ttt_main.cpp ./tictactoe.cpp ./replaymemory.cpp ./dqn.cpp ./sample.cpp ../../lib/math/matrix.cpp ../../lib/ml/nn.cpp -o ttt_main
  // -O1: basic optimizations, minimal compile time impact

  3. g++ -std=c++20 -O2 -I../../lib ttt_main.cpp ./tictactoe.cpp ./replaymemory.cpp ./dqn.cpp ./sample.cpp ../../lib/math/matrix.cpp ../../lib/ml/nn.cpp -o ttt_main
  // -O2: more aggressive optimizations, no space-speed tradeoffs

  4. g++ -std=c++20 -O3 -I../../lib ttt_main.cpp ./tictactoe.cpp ./replaymemory.cpp ./dqn.cpp ./sample.cpp ../../lib/math/matrix.cpp ../../lib/ml/nn.cpp -o ttt_main
  // -O3: function inlining, vectorization, loop unrolling, constant propagation

  5. g++ -std=c++20 -O3 -ffast-math -I../../lib ttt_main.cpp ./tictactoe.cpp ./replaymemory.cpp ./dqn.cpp ./sample.cpp ../../lib/math/matrix.cpp ../../lib/ml/nn.cpp -o ttt_main_fastmath
  // -O3: all O3 optimizations
  // -ffast-math: floating point math optimizations, relaxed IEEE compliance

  6. g++ -std=c++20 -O3 -ffast-math -funroll-loops -flto -march=native -mtune=native -fomit-frame-pointer -I../../lib ttt_main.cpp ./tictactoe.cpp ./replaymemory.cpp ./dqn.cpp ./sample.cpp ../../lib/math/matrix.cpp ../../lib/ml/nn.cpp -o ttt_main
  // -O3: all O3 optimizations
  // -ffast-math: floating point math optimizations, relaxed IEEE compliance
  // -funroll-loops: unroll loops
  // -flto: link time optimizations
  // -march=native: CPU-specific instruction set
  // -mtune=native: CPU-specific scheduling and optimizations
*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>
#include <cstdint>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <thread>

#include "dqn.hpp"
#include "tictactoe.hpp"
#include "sample.hpp"
#include "ml/nn.hpp"
#include "math/matrix.hpp"

#define NUM_GAMES 10000

#define NUM_EPOCHS_BETWEEN_SAVING_MODEL 10

#define PRINT(x) std::cout << x << std::endl;

struct DQNConfig {
    std::string epsilon_decay;
    std::string epsilon_start;
    std::string epsilon_end;
    std::string gamma;
    std::string update_frequency;
    std::string learning_rate;
    std::string batch_size;
    std::function<ml::Sequential()> create_network;
    int num_training_episodes;
    std::string architecture_string;
    int board_dimension = 3; // default to 3x3
    /** 0 = seed from std::random_device; otherwise fixed seed for reproducible training/eval RNG. */
    std::uint64_t training_rng_seed = 0;
};

struct ExperimentConfig {
    std::string experiment_name;
    std::vector<DQNConfig> configs;
    // a lambda for getting the folder name for an experiment - takes as input a DQNConfig and returns a string
    std::function<std::string(DQNConfig)> get_folder_name;
    int board_dim = 3; // default to 3x3
    /** Applied when a DQNConfig has training_rng_seed == 0 (e.g. sweep baseline). */
    std::uint64_t default_training_rng_seed = 0;
};

namespace {

int dqn_state_dim(int board_dim) { return 2 * board_dim * board_dim; }

int dqn_action_dim(int board_dim) { return board_dim * board_dim; }

std::mt19937 make_training_rng(std::uint64_t seed) {
    if (seed == 0) {
        std::random_device rd;
        return std::mt19937(rd());
    }
    std::uint32_t mixed = static_cast<std::uint32_t>(seed ^ (seed >> 32));
    if (mixed == 0) {
        mixed = 1;
    }
    return std::mt19937(mixed);
}

ml::Sequential make_dqn_two_block_mlp(int in_dim, int hidden, int out_dim) {
    ml::Sequential net;
    net.add_layer(std::make_shared<ml::LinearLayer>(in_dim, hidden));
    net.add_layer(std::make_shared<ml::ReLULayer>());
    net.add_layer(std::make_shared<ml::LinearLayer>(hidden, hidden));
    net.add_layer(std::make_shared<ml::ReLULayer>());
    net.add_layer(std::make_shared<ml::LinearLayer>(hidden, out_dim));
    return net;
}

ml::Sequential make_dqn_three_block_mlp(int in_dim, int hidden, int out_dim) {
    ml::Sequential net;
    net.add_layer(std::make_shared<ml::LinearLayer>(in_dim, hidden));
    net.add_layer(std::make_shared<ml::ReLULayer>());
    net.add_layer(std::make_shared<ml::LinearLayer>(hidden, hidden));
    net.add_layer(std::make_shared<ml::ReLULayer>());
    net.add_layer(std::make_shared<ml::LinearLayer>(hidden, hidden));
    net.add_layer(std::make_shared<ml::ReLULayer>());
    net.add_layer(std::make_shared<ml::LinearLayer>(hidden, out_dim));
    return net;
}

/** Matches comment style: in -> h, ReLU, then (Linear h->h, ReLU) repeated relu_sections times, Linear h->out. */
ml::Sequential make_dqn_stacked_hidden_mlp(int in_dim, int hidden, int relu_sections, int out_dim) {
    ml::Sequential net;
    net.add_layer(std::make_shared<ml::LinearLayer>(in_dim, hidden));
    net.add_layer(std::make_shared<ml::ReLULayer>());
    for (int s = 0; s < relu_sections; ++s) {
        net.add_layer(std::make_shared<ml::LinearLayer>(hidden, hidden));
        net.add_layer(std::make_shared<ml::ReLULayer>());
    }
    net.add_layer(std::make_shared<ml::LinearLayer>(hidden, out_dim));
    return net;
}

} // namespace

/**
 * Default 3x3 DQN MLP: state 18 -> two 32-wide hidden blocks -> 9 actions.
 */
ml::Sequential create_network() {
    const int bd = 3;
    return make_dqn_two_block_mlp(dqn_state_dim(bd), 32, dqn_action_dim(bd));
}

void validate_config(const DQNConfig& config) {
    if (config.epsilon_decay.empty()) {
        std::cerr << "DQNConfig: " << config.architecture_string << std::endl;
        throw std::runtime_error("epsilon_decay is empty!");
    }
    if (config.gamma.empty()) {
        std::cerr << "DQNConfig: " << config.architecture_string << std::endl; 
        throw std::runtime_error("gamma is empty!");
    }
    if (config.epsilon_start.empty()) {
        std::cerr << "DQNConfig: " << config.architecture_string << std::endl;
        throw std::runtime_error("epsilon_start is empty!");
    }
    if (config.epsilon_end.empty()) {
        std::cerr << "DQNConfig: " << config.architecture_string << std::endl;
        throw std::runtime_error("epsilon_end is empty!");
    }
    if (config.update_frequency.empty()) {
        std::cerr << "DQNConfig: " << config.architecture_string << std::endl;
        throw std::runtime_error("update_frequency is empty!");
    }
    if (config.learning_rate.empty()) {
        std::cerr << "DQNConfig: " << config.architecture_string << std::endl;
        throw std::runtime_error("learning_rate is empty!");
    }
    if (config.batch_size.empty()) {
        std::cerr << "DQNConfig: " << config.architecture_string << std::endl;
        throw std::runtime_error("batch_size is empty!");
    }
}

DQNAgent create_agent(DQNConfig config, std::string file_path, bool eval_mode, int board_dim) {
    (void)board_dim; // reserved for shape checks vs. saved weights
    ml::Sequential q_net = config.create_network();
    ml::Sequential target_net = config.create_network();

    validate_config(config);

    DQNAgent agent(q_net, target_net, std::stof(config.epsilon_start), std::stof(config.epsilon_end),
                   std::stof(config.epsilon_decay), std::stof(config.gamma), std::stof(config.learning_rate),
                   std::stoi(config.batch_size), 10000, std::stoi(config.update_frequency));
    agent.load(file_path);
    if (eval_mode) {
        agent.set_epsilon(0.0f);
    }
    return agent;
}
DQNAgent train_agent(DQNConfig config, std::function<std::string(DQNConfig)> get_folder_name, int board_dim,
                     std::uint64_t experiment_default_rng_seed = 0) {
    
    ml::Sequential q_net = config.create_network();
    ml::Sequential target_net = config.create_network();

    int batch_size = std::stoi(config.batch_size);

    validate_config(config);

    DQNAgent agent(q_net, target_net, std::stof(config.epsilon_start), std::stof(config.epsilon_end),
                   std::stof(config.epsilon_decay), std::stof(config.gamma), std::stof(config.learning_rate),
                   std::stoi(config.batch_size), 10000, std::stoi(config.update_frequency));
    
    const int train_board = board_dim;
    TicTacToe env(train_board);

    std::cout << "board dimension: " << train_board << std::endl;

    std::uint64_t seed = config.training_rng_seed;
    if (seed == 0 && experiment_default_rng_seed != 0) {
        seed = experiment_default_rng_seed;
    }
    std::mt19937 gen = make_training_rng(seed);
    std::uniform_int_distribution<> dis(0, 1);
    
    std::string folder_name = get_folder_name(config);

    PRINT("folder name: " << folder_name);

    std::string mkdir_cmd = "mkdir -p \"" + folder_name + "\"";
    system(mkdir_cmd.c_str());

    for (int episode = 0; episode < config.num_training_episodes; ++episode) {
        env.reset();
        std::vector<float> state = env.get_state();
        bool done = false;
        
        bool agent_turn = true;
        
        while (!done) {
            std::vector<int> valid_actions = env.get_valid_actions();
            
            if (agent_turn) {
                int action = agent.act(state, valid_actions);
                auto [next_state, reward] = env.step(action);
                done = env.is_game_over();

                Sample s(state, action, reward, next_state, done);
                agent.add_to_memory(s);
                state = next_state;
            } else {
                std::uniform_int_distribution<> action_dis(0, valid_actions.size() - 1);
                int action = valid_actions[action_dis(gen)];
                auto [next_state, reward] = env.step(action);
                done = env.is_game_over();
                state = next_state;
            }
            
            agent_turn = !agent_turn;
            agent.update_networks(batch_size);
        }
        
        agent.decay_epsilon();

        if ((episode + 1) % 100 == 0) {
            std::cout << "Episode: " << episode + 1 << ", Epsilon: " << agent.get_epsilon() << std::endl;
        }
                if ((episode + 1) % NUM_EPOCHS_BETWEEN_SAVING_MODEL == 0) {
            std::string model_path = folder_name + "/dqn_tictactoe_" + std::to_string(episode + 1) + ".model";
            agent.save(model_path);
        }
    }

    return agent;
}

/**
 * Get the episode number from the file name
 * The format of the input should be `dqn_tictactoe_<episode_number>.model`
 */
int episode_number_from_file_name(std::string file_name) {
    size_t last_underscore = file_name.find_last_of('_');
    if (last_underscore == std::string::npos) {
        return -1; 
    }

    size_t dot_pos = file_name.find(".model", last_underscore);
    if (dot_pos == std::string::npos) {
        return -1;
    }

    std::string number_str = file_name.substr(last_underscore + 1, dot_pos - last_underscore - 1);
    
    try {
        return std::stoi(number_str);
    }
    catch (const std::exception&) {
        return -1;
    }
}

std::vector<int> test_agent_against_random(DQNConfig config, std::string file_path, bool eval_mode, int board_dim,
                                           std::uint64_t experiment_default_rng_seed = 0) {
    DQNAgent agent = create_agent(config, file_path, eval_mode, board_dim);
    
    TicTacToe env(board_dim); // n x n board
    std::uint64_t seed = config.training_rng_seed;
    if (seed == 0 && experiment_default_rng_seed != 0) {
        seed = experiment_default_rng_seed;
    }
    std::mt19937 gen = make_training_rng(seed);
    
    int agent_wins = 0;
    int random_wins = 0;
    int draws = 0;
    
    for (int game = 0; game < NUM_GAMES; ++game) {
        env.reset();
        std::vector<float> state = env.get_state();
        bool done = false;
        bool agent_turn = true;
        
        while (!done) {
            std::vector<int> valid_actions = env.get_valid_actions();
            
            if (agent_turn) {
                int action = agent.act(state, valid_actions);
                auto [next_state, reward] = env.step(action);
                done = env.is_game_over();
                state = next_state;
            } else {
                std::uniform_int_distribution<> action_dis(0, valid_actions.size() - 1);
                int action = valid_actions[action_dis(gen)];
                auto [next_state, reward] = env.step(action);
                done = env.is_game_over();
                state = next_state;
            }
            
            agent_turn = !agent_turn;
        }
        
        int winner = env.get_winner();
        if (winner == 1) {
            agent_wins++;
        }
        else if (winner == 2) {
            random_wins++;
        }
        else {
            draws++;
        }
    }
    return {agent_wins, random_wins, draws};
}

enum RandomGameResult {
    PLAYER_WINS,
    AGENT_WINS,
    DRAW
};

RandomGameResult play_against_random(ml::Sequential net, int board_dimension) {
    ml::Sequential target_net;
    DQNAgent agent(net, target_net, 0.99f, 0.1f, 0.99995f, 0.9f, 0.001f, 64);
    agent.set_epsilon(0.0f);
    TicTacToe env(board_dimension);
    env.reset();
    bool done = false;
    bool agent_turn = true;

    std::cout << "Starting game between agent (X) and random player (O)" << std::endl;
    std::cout << "Initial board state:" << std::endl;
    env.render();
    
    while (!done) {
        std::vector<int> valid_actions = env.get_valid_actions();
        int action;
        if (agent_turn) {
            std::vector<float> state = env.get_state();
            action = agent.act(state, valid_actions);
            std::cout << "\nAgent plays position: " << action << std::endl;
        } else {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, valid_actions.size() - 1);
            action = valid_actions[dis(gen)];
            std::cout << "\nRandom player plays position: " << action << std::endl;
        }
        auto [next_state, reward] = env.step(action);
        done = env.is_game_over();
        agent_turn = !agent_turn;
        
        std::cout << "Current board state:" << std::endl;
        env.render();
    }

    int winner = env.get_winner();
    std::cout << "\nGame Over! ";
    if (winner == 1) {
        std::cout << "Agent wins!" << std::endl;
        return RandomGameResult::AGENT_WINS;
    }
    else if (winner == 2) {
        std::cout << "Random player wins!" << std::endl;
        return RandomGameResult::PLAYER_WINS;
    }
    else {
        std::cout << "It's a draw!" << std::endl;
        return RandomGameResult::DRAW;
    }
}

void play_many_random_games(ml::Sequential net, int board_dimension, int num_games) {
    std::vector<RandomGameResult> results;
    for (int i = 0; i < num_games; ++i) {
        results.push_back(play_against_random(net, board_dimension));
    }

    // compute the number of wins, draws, and losses, and also the percentages of each of those

    int agent_wins = 0;
    int player_wins = 0;
    int draws = 0;

    for (const auto& result : results) {
        if (result == RandomGameResult::AGENT_WINS) agent_wins++;
        else if (result == RandomGameResult::PLAYER_WINS) player_wins++;
        else draws++;
    }

    float agent_win_percentage = static_cast<float>(agent_wins) / num_games;
    float player_win_percentage = static_cast<float>(player_wins) / num_games;
    float draw_percentage = static_cast<float>(draws) / num_games;

    std::cout << "Agent wins: " << agent_wins << " (" << agent_win_percentage << "%)" << std::endl;
    std::cout << "Player wins: " << player_wins << " (" << player_win_percentage << "%)" << std::endl;
    std::cout << "Draws: " << draws << " (" << draw_percentage << "%)" << std::endl;
   
}

std::map<int, std::vector<int>> test_all_models_for_epsilon_decay(DQNConfig config, std::vector<std::string> folder_files, int board_dim,
                                                                  std::uint64_t experiment_default_rng_seed = 0) {
    // map the number of episodes trained to the results of that model
    std::map<int, std::vector<int>> results;

    for (std::string file_name : folder_files) {
        PRINT("Testing " << file_name);
        int num_episodes = episode_number_from_file_name(file_name);
        std::vector<int> result = test_agent_against_random(config, file_name, true, board_dim, experiment_default_rng_seed);
        results[num_episodes] = result;
    }
    
    return results;
}

std::vector<std::string> get_model_files(const std::string& directory) {
    std::vector<std::string> files;
    
    #ifdef _WIN32
    std::string command = "dir /b ";
    #else
    std::string command = "ls ";
    #endif
    
    FILE* pipe = popen((command + "\"" + directory + "\"").c_str(), "r");
    if (pipe) {
        char buffer[128];
        while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
            std::string filename = buffer;
            if (filename.back() == '\n') filename.pop_back();
            files.push_back(directory + "/" + filename);
        }
        pclose(pipe);
    }
    
    return files;
}

void save_results_to_csv(const std::map<int, std::vector<int>>& results, 
                        const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    file << "episode_number,agent_wins,random_wins,draws\n";
    
    for (const auto& [episode_number, result] : results) {
        file << episode_number << "," 
             << result[0] << "," 
             << result[1] << "," 
             << result[2] << std::endl;
    }
    
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

void run_epsilon_decay_experiment() {
    // Define epsilon decay values to test as strings. We define them as strings so that we don't get floating point precision issues
    std::vector<std::string> epsilon_decay_values = {
        "0.5",
        "0.7",
        "0.8",
        "0.9",
        "0.97",
        "0.98",
        "0.99",
        "0.99995"
    };

    std::vector<DQNConfig> configs;
    
    for (const auto& epsilon_decay : epsilon_decay_values) {
        DQNConfig config;
        config.batch_size = "64";
        config.learning_rate = "0.0001";
        config.update_frequency = "10";
        config.epsilon_start = "0.9";
        config.epsilon_end = "0.001";
        config.gamma = "0.9";
        config.epsilon_decay = epsilon_decay;
        config.create_network = create_network;
        config.num_training_episodes = 10000;
        
        configs.push_back(config);
    }
    
    std::map<std::string, std::string> epsilon_decay_to_folder;
    for (const std::string& epsilon_decay : epsilon_decay_values) {
        epsilon_decay_to_folder[epsilon_decay] = "epsilon = " + epsilon_decay;
    }
    for (const auto& [epsilon_decay_str, folder] : epsilon_decay_to_folder) {
        PRINT("Epsilon decay value: " << epsilon_decay_str << " -> Folder: " << folder);
    }

    // Train the models for each config
    for (const auto& config : configs) {
        train_agent(config, [](DQNConfig config) {
            return "epsilon = " + config.epsilon_decay;
        }, 3);
    }
    
    for (const auto& config : configs) {
        std::vector<std::string> model_files = get_model_files("epsilon = " + config.epsilon_decay);
        for (const auto& model_file : model_files) {
            PRINT("Model file: " << model_file);
        }

        std::map<int, std::vector<int>> results = test_all_models_for_epsilon_decay(config, model_files, 3);
        save_results_to_csv(results, config.epsilon_decay + " results.csv");
    }
    

}

void run_gamma_experiment() {
    std::vector<std::string> gamma_values = {
        "0.2",
        "0.4",
        "0.6",
        "0.8",
        "0.9",
        "0.95",
        "0.98",
        "0.99995"
    };

    std::string experiment_folder = "gamma";
    std::string mkdir_cmd = "mkdir -p \"" + experiment_folder + "\"";
    system(mkdir_cmd.c_str());

    std::vector<DQNConfig> configs;
    DQNConfig base_config;
    base_config.epsilon_decay = "0.99995";
    base_config.epsilon_start = "0.9";
    base_config.epsilon_end = "0.001";
    base_config.update_frequency = "10";
    base_config.learning_rate = "0.001";
    base_config.batch_size = "64";
    base_config.num_training_episodes = 10000;

    for (const auto& gamma_value : gamma_values) {
        DQNConfig config = base_config;
        config.gamma = gamma_value;
        config.create_network = create_network;
        config.num_training_episodes = 10000;
        configs.push_back(config);
    }

    // Train all models, saving in gamma/0, gamma/1, ...
    for (size_t i = 0; i < configs.size(); ++i) {
        std::string config_folder = experiment_folder + "/gamma=" + configs[i].gamma;
        train_agent(configs[i], [config_folder](DQNConfig) { return config_folder; }, 3);
    }

    // Evaluate all models and save results in gamma/gamma=0.5.csv, gamma/gamma=0.7.csv, ...
    for (size_t i = 0; i < configs.size(); ++i) {
        std::string config_folder = experiment_folder + "/gamma=" + configs[i].gamma;
        std::vector<std::string> model_files = get_model_files(config_folder);
        std::map<int, std::vector<int>> results = test_all_models_for_epsilon_decay(configs[i], model_files, 3);
        save_results_to_csv(results, experiment_folder + "/gamma=" + configs[i].gamma + "_results.csv");
    }
}

ExperimentConfig epsilon_decay_experiment(int board_dim = 3) {
    ExperimentConfig experiment_config;
    experiment_config.experiment_name = "epsilon_decay";
    experiment_config.get_folder_name = [](DQNConfig config) {
        return "epsilon_decay=" + config.epsilon_decay;
    };
    experiment_config.board_dim = board_dim;
    experiment_config.default_training_rng_seed = 42;
    DQNConfig base_config;
    base_config.gamma = "0.99995";
    base_config.epsilon_start = "0.9";
    base_config.epsilon_end = "0.001";
    base_config.update_frequency = "10";
    base_config.learning_rate = "0.001";
    base_config.batch_size = "64";
    base_config.num_training_episodes = 10000;
    std::vector<std::string> epsilon_decay_values = {
        "0.9", "0.95", "0.98", "0.99", "0.995", "0.998", "0.999", "0.9995", "0.9998", "0.99995"
    };
    for (const auto& epsilon_decay_value : epsilon_decay_values) {
        DQNConfig config = base_config;
        config.epsilon_decay = epsilon_decay_value;
        config.create_network = create_network;
        config.board_dimension = board_dim;
        config.num_training_episodes = 10000;
        experiment_config.configs.push_back(config);
    }
    return experiment_config;
}

ExperimentConfig gamma_experiment() {
    ExperimentConfig experiment_config;
    experiment_config.experiment_name = "gamma";
    experiment_config.get_folder_name = [](DQNConfig config) {
        return "gamma=" + config.gamma;
    };

    DQNConfig base_config;
    base_config.epsilon_decay = "0.99995";
    base_config.epsilon_start = "0.9";
    base_config.epsilon_end = "0.001";
    base_config.update_frequency = "10";
    base_config.learning_rate = "0.001";
    base_config.batch_size = "64";
    base_config.num_training_episodes = 10000;
    experiment_config.board_dim = 3;
    experiment_config.default_training_rng_seed = 42;

    std::vector<std::string> gamma_values = {
        "0.2",
        "0.4",
        "0.6",
        "0.8",
        "0.9",
        "0.95",
        "0.98",
        "0.99995"
    };

    for (const auto& gamma_value : gamma_values) {
        DQNConfig config = base_config;
        config.gamma = gamma_value;
        config.create_network = create_network;
        config.board_dimension = experiment_config.board_dim;
        config.num_training_episodes = 10000;
        experiment_config.configs.push_back(config);
    }

    return experiment_config;
}

void run_experiment(ExperimentConfig experiment_config) {
    std::string mkdir_cmd = "mkdir -p \"" + experiment_config.experiment_name + "\"";
    system(mkdir_cmd.c_str());

    // train the models
    for (DQNConfig config : experiment_config.configs) {
        std::string config_folder = experiment_config.experiment_name + "/" + experiment_config.get_folder_name(config);
        train_agent(config, [config_folder](DQNConfig) { return config_folder; }, experiment_config.board_dim,
                      experiment_config.default_training_rng_seed);
    }

    // evaluate the models and output the performance results to a CSV files
    for (DQNConfig config : experiment_config.configs) {
        std::vector<std::string> model_files = get_model_files(experiment_config.experiment_name + "/" + experiment_config.get_folder_name(config));
        std::map<int, std::vector<int>> results = test_all_models_for_epsilon_decay(config, model_files, experiment_config.board_dim,
                                                                                     experiment_config.default_training_rng_seed);
        save_results_to_csv(results, experiment_config.experiment_name + "/" + experiment_config.get_folder_name(config) + "_results.csv");
    }
    
}

ExperimentConfig learning_rate_experiment() {

   ExperimentConfig experiment_config;
   experiment_config.experiment_name = "learning_rate";
   experiment_config.get_folder_name = [](DQNConfig config) {
        return "learning_rate=" + config.learning_rate;
   };
   experiment_config.board_dim = 3;
   experiment_config.default_training_rng_seed = 42;

   DQNConfig base_config;
   base_config.epsilon_decay = "0.9";
   base_config.epsilon_start = "0.9";
   base_config.epsilon_end = "0.001";
   base_config.update_frequency = "10";
   base_config.batch_size = "64";
   base_config.gamma = "0.9";
   base_config.learning_rate = "0.0001";
   base_config.num_training_episodes = 10000;
   
   std::vector<std::string> learning_rate_values = {
    "0.00001",
    "0.00005",
    "0.0001",
    "0.0005",
    "0.001",
    "0.005",
    "0.01",
    "0.05",
    "0.1"
   };

   for (std::string learning_rate : learning_rate_values) {
    DQNConfig config = base_config;
    config.learning_rate = learning_rate;
    config.create_network = create_network;
    config.board_dimension = experiment_config.board_dim;
    config.num_training_episodes = 10000;
    experiment_config.configs.push_back(config);
   }

   return experiment_config;
}

ExperimentConfig dummy_experiment() {
    ExperimentConfig experiment_config;
    experiment_config.experiment_name = "dummy_experiment";
    experiment_config.get_folder_name = [](DQNConfig config) {
        return "dummy_experiment = " + config.epsilon_decay;
    };
    experiment_config.board_dim = 3;
    experiment_config.default_training_rng_seed = 42;

    DQNConfig config;
    config.epsilon_decay = "0.99995";
    config.epsilon_start = "0.9";
    config.epsilon_end = "0.001";
    config.update_frequency = "10";
    config.learning_rate = "0.001";
    config.batch_size = "64";
    config.gamma = "0.9";
    config.create_network = create_network;
    config.num_training_episodes = 1000;
    
    std::vector<std::string> epsilon_decay_values = {
        "0.5",
        "0.7", 
        "0.8",
        "0.9",
        "0.97",
    };

    for (std::string epsilon_decay : epsilon_decay_values) {
        DQNConfig trial = config;
        trial.epsilon_decay = epsilon_decay;
        trial.board_dimension = experiment_config.board_dim;
        experiment_config.configs.push_back(trial);
    }

    return experiment_config;
}

ExperimentConfig epsilon_minimum_value_experiment() {
    ExperimentConfig experiment_config;
    experiment_config.experiment_name = "epsilon_minimum_value";
    experiment_config.get_folder_name = [](DQNConfig config) {
        return "epsilon_minimum_value=" + config.epsilon_end;
    };
    experiment_config.board_dim = 3;
    experiment_config.default_training_rng_seed = 42;

    DQNConfig base_config;
    base_config.epsilon_decay = "0.99995";
    base_config.epsilon_start = "0.9";
    base_config.epsilon_end = "0.001";
    base_config.update_frequency = "10";
    base_config.learning_rate = "0.0001";
    base_config.batch_size = "64";
    base_config.gamma = "0.9";
    base_config.create_network = create_network;
    base_config.num_training_episodes = 10000;

    std::vector<std::string> epsilon_end_values = {
        "0.01",
        "0.05",
        "0.1",
        "0.2",
        "0.3",
        "0.4",
        "0.5",
        "0.75"
    };

    for (std::string epsilon_end : epsilon_end_values) {
        DQNConfig cfg = base_config;
        cfg.epsilon_end = epsilon_end;
        cfg.board_dimension = experiment_config.board_dim;
        experiment_config.configs.push_back(cfg);
    }
    
    return experiment_config;
}

ExperimentConfig epsilon_start_value_experiment() {
    ExperimentConfig experiment_config;
    experiment_config.experiment_name = "epsilon_start_value";
    experiment_config.get_folder_name = [](DQNConfig config) {
        return "epsilon_start_value=" + config.epsilon_start;
    };
    experiment_config.board_dim = 3;
    experiment_config.default_training_rng_seed = 42;

    DQNConfig base_config;
    base_config.epsilon_decay = "0.9";
    base_config.epsilon_end = "0.001";
    base_config.update_frequency = "10";
    base_config.learning_rate = "0.0001";
    base_config.batch_size = "64";
    base_config.gamma = "0.9";
    base_config.create_network = create_network;
    base_config.num_training_episodes = 10000;

    std::vector<std::string> epsilon_start_values = {
        "0.99",
        "0.9",
        "0.7",
        "0.4",
        "0.2",
        "0.05",
        "0.01"
    };

    for (std::string epsilon_start : epsilon_start_values) {
        DQNConfig cfg = base_config;
        cfg.epsilon_start = epsilon_start;
        cfg.create_network = create_network;
        cfg.board_dimension = experiment_config.board_dim;
        cfg.num_training_episodes = 10000;
        experiment_config.configs.push_back(cfg);
    }

    return experiment_config;
}

ExperimentConfig batch_size_experiment() {
    ExperimentConfig experiment_config;
    experiment_config.experiment_name = "batch_size";
    experiment_config.get_folder_name = [](DQNConfig config) {
        return "batch_size=" + config.batch_size;
    };
    experiment_config.board_dim = 3;
    experiment_config.default_training_rng_seed = 42;

    DQNConfig base_config;
    base_config.epsilon_decay = "0.999";
    base_config.epsilon_start = "0.9";
    base_config.epsilon_end = "0.001";
    base_config.update_frequency = "10";
    base_config.learning_rate = "0.0001";
    base_config.gamma = "0.9";
    base_config.batch_size = "64";
    base_config.create_network = create_network;
    base_config.num_training_episodes = 10000;

    std::vector<std::string> batch_size_values = {
        "8",
        "16", 
        "32",
        "64",
        "128",
        "256",
        "512"
    };

    for (std::string batch_size : batch_size_values) {
        DQNConfig cfg = base_config;
        cfg.batch_size = batch_size;
        cfg.create_network = create_network;
        cfg.board_dimension = experiment_config.board_dim;
        cfg.num_training_episodes = 10000;
        experiment_config.configs.push_back(cfg);
    }

    return experiment_config;
}

ExperimentConfig update_frequency_experiment() {
    ExperimentConfig experiment_config;
    experiment_config.experiment_name = "update_frequency";
    experiment_config.get_folder_name = [](DQNConfig config) {
        return "update_frequency=" + config.update_frequency;
    };
    experiment_config.board_dim = 3;
    experiment_config.default_training_rng_seed = 42;

    DQNConfig base_config;
    base_config.epsilon_decay = "0.99";
    base_config.epsilon_start = "0.9";
    base_config.epsilon_end = "0.001";
    base_config.update_frequency = "10";
    base_config.learning_rate = "0.0001";
    base_config.batch_size = "512";
    base_config.gamma = "0.9";
    base_config.create_network = create_network;
    base_config.num_training_episodes = 10000;

    std::vector<std::string> update_frequency_values = {
        "1",
        "5",
        "10",
        "25",
        "50",
        "100"
    };

    for (std::string update_frequency : update_frequency_values) {
        DQNConfig cfg = base_config;
        cfg.update_frequency = update_frequency;
        cfg.board_dimension = experiment_config.board_dim;
        experiment_config.configs.push_back(cfg);
    }

    return experiment_config;
}


ExperimentConfig architecture_experiment() {
    /*
        Widths and depths match the intended sweep; input/output dims follow board size
        (e.g. 4x4 => state 32, actions 16).
    */

    ExperimentConfig experiment_config;
    experiment_config.experiment_name = "architecture";
    experiment_config.get_folder_name = [](DQNConfig config) {
        return "architecture=" + config.architecture_string;
    };
    const int bd = 4;
    experiment_config.board_dim = bd;
    experiment_config.default_training_rng_seed = 42;

    const int in_d = dqn_state_dim(bd);
    const int out_d = dqn_action_dim(bd);

    struct ArchSpec {
        std::string label;
        std::function<ml::Sequential()> factory;
    };

    const std::vector<ArchSpec> archs = {
        {std::to_string(in_d) + " -> 64 -> ReLU -> 64 -> ReLU -> " + std::to_string(out_d),
         [=]() { return make_dqn_two_block_mlp(in_d, 64, out_d); }},
        {std::to_string(in_d) + " -> 32 -> ReLU -> 32 -> ReLU -> " + std::to_string(out_d),
         [=]() { return make_dqn_two_block_mlp(in_d, 32, out_d); }},
        {std::to_string(in_d) + " -> 16 -> ReLU -> 16 -> ReLU -> " + std::to_string(out_d),
         [=]() { return make_dqn_two_block_mlp(in_d, 16, out_d); }},
        {std::to_string(in_d) + " -> 8 -> ReLU -> 8 -> ReLU -> " + std::to_string(out_d),
         [=]() { return make_dqn_two_block_mlp(in_d, 8, out_d); }},
        {std::to_string(in_d) + " -> 4 -> ReLU -> 4 -> ReLU -> " + std::to_string(out_d),
         [=]() { return make_dqn_two_block_mlp(in_d, 4, out_d); }},

        {std::to_string(in_d) + " -> 16 -> ReLU -> 16 -> ReLU -> 16 -> ReLU -> " + std::to_string(out_d),
         [=]() { return make_dqn_three_block_mlp(in_d, 16, out_d); }},
        {std::to_string(in_d) + " -> 8 -> ReLU -> (8)x3 -> ReLU -> " + std::to_string(out_d),
         [=]() { return make_dqn_stacked_hidden_mlp(in_d, 8, 3, out_d); }},
        {std::to_string(in_d) + " -> 4 -> ReLU -> (4)x3 -> ReLU -> " + std::to_string(out_d),
         [=]() { return make_dqn_stacked_hidden_mlp(in_d, 4, 3, out_d); }},
    };

    DQNConfig base_config;
    base_config.epsilon_decay = "0.99";
    base_config.epsilon_start = "0.9";
    base_config.epsilon_end = "0.001";
    base_config.update_frequency = "10";
    base_config.learning_rate = "0.0001";
    base_config.batch_size = "64";
    base_config.num_training_episodes = 10000;
    base_config.gamma = "0.9";
    base_config.board_dimension = bd;

    for (const ArchSpec& spec : archs) {
        DQNConfig config = base_config;
        config.architecture_string = spec.label;
        config.create_network = spec.factory;
        experiment_config.configs.push_back(config);
    }

    return experiment_config;
}

ExperimentConfig reproduction_experiment() {
    // this experiment checks how reproducible the results are when training the same model multiple times

    ExperimentConfig experiment_config;
    experiment_config.experiment_name = "reproduction";
    experiment_config.get_folder_name = [](DQNConfig config) {
        return "trial=" + config.architecture_string;
    };
    experiment_config.board_dim = 3;
    experiment_config.default_training_rng_seed = 0;

    DQNConfig base_config;

    base_config.epsilon_decay = "0.999";
    base_config.epsilon_start = "0.9";
    base_config.epsilon_end = "0.001";
    base_config.update_frequency = "10";
    base_config.learning_rate = "0.0001";
    base_config.batch_size = "64";
    base_config.gamma = "0.9";
    base_config.create_network = create_network;
    base_config.architecture_string = "0";
    base_config.num_training_episodes = 10000;

    std::vector<std::string> architecture_strings;

    for (int i = 1; i <= 10; i++) {
        architecture_strings.push_back(std::to_string(i));
    }

    for (std::string architecture_string : architecture_strings) {
        DQNConfig cfg = base_config;
        cfg.architecture_string = architecture_string;
        cfg.training_rng_seed = 900000u + static_cast<std::uint64_t>(std::stoul(architecture_string));
        cfg.board_dimension = experiment_config.board_dim;
        experiment_config.configs.push_back(cfg);
    }

    return experiment_config;
}

ExperimentConfig train_single_model_config() {
    // same as the experiment above but just train one time
    ExperimentConfig experiment_config;
    experiment_config.experiment_name = "train_single_model";
    experiment_config.get_folder_name = [](DQNConfig config) {
        return "trial=" + config.architecture_string;
    };
    experiment_config.board_dim = 3;
    experiment_config.default_training_rng_seed = 42;
    
    DQNConfig base_config;
    base_config.epsilon_decay = "0.999";
    base_config.epsilon_start = "0.9";
    base_config.epsilon_end = "0.001";
    base_config.update_frequency = "10";
    base_config.learning_rate = "0.0001";
    base_config.batch_size = "64";
    base_config.gamma = "0.9";
    base_config.create_network = create_network;
    base_config.architecture_string = "1";
    base_config.board_dimension = experiment_config.board_dim;
    base_config.num_training_episodes = 100;

    experiment_config.configs.push_back(base_config);

    return experiment_config;
}

ExperimentConfig random_grid_experiment() {
    ExperimentConfig experiment_config;

    experiment_config.experiment_name = "random_grid";
    experiment_config.get_folder_name = [](DQNConfig config) {
        return "random_grid=" + config.epsilon_decay + "_" + config.epsilon_start + "_" + config.epsilon_end + "_" + config.update_frequency + "_" + config.learning_rate + "_" + config.batch_size + "_" + config.architecture_string;
    };

    DQNConfig base_config;
    base_config.epsilon_decay = "0.99";
    base_config.epsilon_start = "0.9";
    base_config.epsilon_end = "0.001";
    base_config.update_frequency = "10";
    base_config.learning_rate = "0.0001";
    base_config.batch_size = "64";
    base_config.gamma = "0.9";
    base_config.create_network = create_network;
    base_config.architecture_string = "18 -> 32 -> ReLU -> 32 -> ReLU -> 9";
    base_config.num_training_episodes = 10000;

   throw std::runtime_error("Not implemented");
}

float train_single_model_experiment() {
    ExperimentConfig experiment_config = train_single_model_config();

    // save current time

    auto now = std::chrono::system_clock::now();
    auto start_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count();

    run_experiment(experiment_config);

    auto end_time = std::chrono::system_clock::now();
    auto end_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time.time_since_epoch()
    ).count();

    return end_milliseconds - start_milliseconds;
}

ExperimentConfig four_by_four_experiment_one() {
    
    DQNConfig config;
    config.epsilon_decay = "0.999";
    config.epsilon_start = "0.9";
    config.epsilon_end = "0.001";
    config.update_frequency = "10";
    config.learning_rate = "0.0001";
    config.batch_size = "64";
    config.gamma = "0.9";
    config.board_dimension = 4;
    config.num_training_episodes = 30000;
    config.create_network = []() {
        ml::Sequential net;
        net.add_layer(std::make_shared<ml::LinearLayer>(32, 64));
        net.add_layer(std::make_shared<ml::ReLULayer>());
        net.add_layer(std::make_shared<ml::LinearLayer>(64, 64));
        net.add_layer(std::make_shared<ml::ReLULayer>());
        net.add_layer(std::make_shared<ml::LinearLayer>(64, 64));
        net.add_layer(std::make_shared<ml::ReLULayer>());
        net.add_layer(std::make_shared<ml::LinearLayer>(64, 16));
        return net;
    };

    config.architecture_string = "32 -> 64 -> ReLU -> 64 -> ReLU -> 64 -> ReLU -> 64 -> 16";

    ExperimentConfig experiment_config;

    experiment_config.experiment_name = "4x4 good architecture";
    experiment_config.get_folder_name = [](DQNConfig) {
        return "4x4";
    };

    experiment_config.board_dim = 4;
    experiment_config.default_training_rng_seed = 42;

    experiment_config.configs.push_back(config);

    return experiment_config;
}

ExperimentConfig four_by_four_experiment_two() {
    
    DQNConfig config;
    config.epsilon_decay = "0.999";
    config.epsilon_start = "0.9";
    config.epsilon_end = "0.001";
    config.update_frequency = "10";
    config.learning_rate = "0.0001";
    config.batch_size = "64";
    config.gamma = "0.9";
    config.board_dimension = 4;
    config.num_training_episodes = 10000;
    config.create_network = []() {
        ml::Sequential net;
        // 32 -> 32 -> ReLU -> 32 -> ReLU -> 32 -> ReLU -> 16
        net.add_layer(std::make_shared<ml::LinearLayer>(32, 32));
        net.add_layer(std::make_shared<ml::ReLULayer>());
        net.add_layer(std::make_shared<ml::LinearLayer>(32, 32));
        net.add_layer(std::make_shared<ml::ReLULayer>());
        net.add_layer(std::make_shared<ml::LinearLayer>(32, 32));
        net.add_layer(std::make_shared<ml::ReLULayer>());
        net.add_layer(std::make_shared<ml::LinearLayer>(32, 16));
        return net;
    };

    config.architecture_string = "32 -> 32 -> ReLU -> 32 -> ReLU -> 32 -> ReLU -> 16";

    ExperimentConfig experiment_config;

    experiment_config.experiment_name = "four_by_four_" + config.architecture_string;
    experiment_config.get_folder_name = [](DQNConfig config) {
        return "four_by_four_" + config.architecture_string;
    };

    experiment_config.board_dim = 4;
    experiment_config.default_training_rng_seed = 42;

    experiment_config.configs.push_back(config);

    return experiment_config;
}

ExperimentConfig four_by_four_experiment_three() {
    
    DQNConfig config;
    config.epsilon_decay = "0.999";
    config.epsilon_start = "0.9";
    config.epsilon_end = "0.001";
    config.update_frequency = "10";
    config.learning_rate = "0.0001";
    config.batch_size = "64";
    config.gamma = "0.9";
    config.board_dimension = 4;
    config.num_training_episodes = 5000;
    config.create_network = []() {
        ml::Sequential net;
        net.add_layer(std::make_shared<ml::LinearLayer>(32, 64));
        net.add_layer(std::make_shared<ml::ReLULayer>());
        net.add_layer(std::make_shared<ml::LinearLayer>(64, 128));
        net.add_layer(std::make_shared<ml::ReLULayer>());
        net.add_layer(std::make_shared<ml::LinearLayer>(128, 64));
        net.add_layer(std::make_shared<ml::ReLULayer>());
        net.add_layer(std::make_shared<ml::LinearLayer>(64, 16));
        return net;
    };

    config.architecture_string = "32 -> 64 -> ReLU -> 128 -> ReLU -> 64 -> ReLU -> 16";

    ExperimentConfig experiment_config;

    experiment_config.experiment_name = "four_by_four_" + config.architecture_string;
    experiment_config.get_folder_name = [](DQNConfig config) {
        return "four_by_four_" + config.architecture_string;
    };

    experiment_config.board_dim = 4;
    experiment_config.default_training_rng_seed = 42;

    experiment_config.configs.push_back(config);

    return experiment_config;
}

ExperimentConfig four_by_four_experiment_four() {
    
    DQNConfig config;
    config.epsilon_decay = "0.999";
    config.epsilon_start = "0.9";
    config.epsilon_end = "0.001";
    config.update_frequency = "10";
    config.learning_rate = "0.0001";
    config.batch_size = "64";
    config.gamma = "0.9";
    config.board_dimension = 4;
    config.num_training_episodes = 5000;
    config.create_network = []() {
        ml::Sequential net;
        net.add_layer(std::make_shared<ml::LinearLayer>(32, 64));
        net.add_layer(std::make_shared<ml::ReLULayer>());
        net.add_layer(std::make_shared<ml::LinearLayer>(64, 64));
        net.add_layer(std::make_shared<ml::ReLULayer>());
        net.add_layer(std::make_shared<ml::LinearLayer>(64, 16));
        return net;
    };

    config.architecture_string = "32 -> 64 -> ReLU -> 64 -> 16";

    ExperimentConfig experiment_config;

    experiment_config.experiment_name = "four_by_four_" + config.architecture_string;
    experiment_config.get_folder_name = [](DQNConfig config) {
        return "four_by_four_" + config.architecture_string;
    };

    experiment_config.board_dim = 4;
    experiment_config.default_training_rng_seed = 42;

    experiment_config.configs.push_back(config);

    return experiment_config;
}


enum PhaseOneOption {
    RUN_EXPERIMENT,
    LOAD_MODEL
};

PhaseOneOption get_phase_one_option() {
    std::string input;
    while (true) {
        std::cout << "Enter 1 to run experiment, 2 to load model: ";
        std::cin >> input;
        if (input == "1") {
            return PhaseOneOption::RUN_EXPERIMENT;
        } else if (input == "2") {
            return PhaseOneOption::LOAD_MODEL;
        }
        std::cout << "Invalid input. Please try again." << std::endl;
    }
}

void run_experiment_phase_one() {

    std::vector<ExperimentConfig> experiment_configs = {
        epsilon_decay_experiment(),
        gamma_experiment(),
        learning_rate_experiment(),
        dummy_experiment(),
        epsilon_minimum_value_experiment(),
        epsilon_start_value_experiment(),
        batch_size_experiment(),
        architecture_experiment(),
        reproduction_experiment(),
        train_single_model_config(),
        four_by_four_experiment_one(),
        four_by_four_experiment_two(),
        four_by_four_experiment_three(),
        four_by_four_experiment_four(),
    };

    for (size_t i = 0; i < experiment_configs.size(); i++) {
        std::cout << (i + 1) << ". " << experiment_configs[i].experiment_name << std::endl;
    }

    std::cout << "Enter the index of the experiment you want to run: ";

    int index;
    
    std::cin >> index;

    if (index < 1 || index > static_cast<int>(experiment_configs.size())) {
        std::cout << "Invalid index. Please try again." << std::endl;
        return run_experiment_phase_one();
    }

    ExperimentConfig experiment_config = experiment_configs[index - 1];
    run_experiment(experiment_config);
}

struct Model {
    std::string architecture_string;
    // lambda that creates the model
    std::function<ml::Sequential()> create_network;
};

enum PhaseTwoOption {
    PLAY_AGAINST_MODEL,
    PLAY_AGAINST_RANDOM
};

PhaseTwoOption get_phase_two_option() {
    std::string input;
    while (true) {
        std::cout << "Enter 1 to play against the model, 2 to play against a random agent: ";
        std::cin >> input;
        if (input == "1") {
            return PhaseTwoOption::PLAY_AGAINST_MODEL;
        } else if (input == "2") {
            return PhaseTwoOption::PLAY_AGAINST_RANDOM;
        }
    }
}

void play_against_model(ml::Sequential net, int board_dimension) {
    bool play_again = true;
    while (play_again) {
        ml::Sequential target_net;
        DQNAgent agent(net, target_net, 0.99f, 0.1f, 0.99995f, 0.9f, 0.001f, 64);
        agent.set_epsilon(0.0f);

        TicTacToe env(board_dimension);
        env.reset();
        bool done = false;

        std::cout << "Playing against loaded model. You are O, agent is X." << std::endl;
        std::cout << "Board positions are numbered 0-" << (board_dimension * board_dimension - 1) << " as follows:" << std::endl;
        for (int i = 0; i < board_dimension; ++i) {
            for (int j = 0; j < board_dimension; ++j) {
                std::cout << std::setw(3) << (i * board_dimension + j) << " ";
            }
            std::cout << std::endl;
            if (i < board_dimension - 1) {
                std::cout << std::string(board_dimension * 4 - 1, '-') << std::endl;
            }
        }
        std::cout << std::endl;

        while (!done) {
            env.render();
            int action;
            if (env.get_current_player() == 1) {
                std::vector<float> state = env.get_state();
                if (state.empty()) {
                    std::cout << "Error: State vector is empty. Aborting." << std::endl;
                    return;
                }
                std::vector<int> valid_actions = env.get_valid_actions();
                action = agent.act(state, valid_actions);
                std::cout << "Agent plays position: " << action << std::endl;
            } else {
                std::vector<int> valid_actions = env.get_valid_actions();
                do {
                    std::cout << "Enter your move (0-" << (board_dimension * board_dimension - 1) << "): ";
                    std::cin >> action;
                    if (action < 0 || action >= board_dimension * board_dimension) {
                        std::cout << "Invalid move! Please enter a number between 0 and " << (board_dimension * board_dimension - 1) << "." << std::endl;
                        continue;
                    }
                } while (std::find(valid_actions.begin(), valid_actions.end(), action) == valid_actions.end());
            }
            auto [next_state, reward] = env.step(action);
            done = env.is_game_over();
        }
        env.render();
        int winner = env.get_winner();
        if (winner == 1) {
            std::cout << "Agent wins!" << std::endl;
        } else if (winner == 2) {
            std::cout << "You win!" << std::endl;
        } else {
            std::cout << "It's a draw!" << std::endl;
        }

        // Ask if player wants to play again
        char response;
        do {
            std::cout << "Would you like to play again? (y/n): ";
            std::cin >> response;
            response = std::tolower(response);
        } while (response != 'y' && response != 'n');
        
        play_again = (response == 'y');
    }
}

void run_load_model_phase_one() {
    std::vector<Model> models = {
        {
            "32 -> 64 -> ReLU -> 64 -> ReLU -> 64 -> ReLU -> 16",
            []() {
                ml::Sequential net;
                net.add_layer(std::make_shared<ml::LinearLayer>(32, 64));
                net.add_layer(std::make_shared<ml::ReLULayer>());
                net.add_layer(std::make_shared<ml::LinearLayer>(64, 64));
                net.add_layer(std::make_shared<ml::ReLULayer>());
                net.add_layer(std::make_shared<ml::LinearLayer>(64, 64));
                net.add_layer(std::make_shared<ml::ReLULayer>());
                net.add_layer(std::make_shared<ml::LinearLayer>(64, 16));
                return net;
            }
        },
        {
            "32 -> 32 -> ReLU -> 32 -> ReLU -> 32 -> ReLU -> 16",
            []() {
                ml::Sequential net;
                net.add_layer(std::make_shared<ml::LinearLayer>(32, 32));
                net.add_layer(std::make_shared<ml::ReLULayer>());
                net.add_layer(std::make_shared<ml::LinearLayer>(32, 32));
                net.add_layer(std::make_shared<ml::ReLULayer>());
                net.add_layer(std::make_shared<ml::LinearLayer>(32, 32));
                net.add_layer(std::make_shared<ml::ReLULayer>());
                net.add_layer(std::make_shared<ml::LinearLayer>(32, 16));
                return net;
            }
        },
        {
            "32 -> 64 -> ReLU -> 128 -> ReLU -> 64 -> ReLU -> 16",
            []() {
                ml::Sequential net;
                net.add_layer(std::make_shared<ml::LinearLayer>(32, 64));
                net.add_layer(std::make_shared<ml::ReLULayer>());
                net.add_layer(std::make_shared<ml::LinearLayer>(64, 128));
                net.add_layer(std::make_shared<ml::ReLULayer>());
                net.add_layer(std::make_shared<ml::LinearLayer>(128, 64));
                net.add_layer(std::make_shared<ml::ReLULayer>());
                net.add_layer(std::make_shared<ml::LinearLayer>(64, 16));
                return net;
            }
        },

    };

    // list the models

    std::cout << "To load a model, first, select the index of the model architecture you want to load:" << std::endl;

    for (size_t i = 0; i < models.size(); i++) {
        std::cout << (i + 1) << ". " << models[i].architecture_string << std::endl;
    }

    int index;

    std::cin >> index;

    if (index < 1 || index > static_cast<int>(models.size())) {
        std::cout << "Invalid index. Please try again." << std::endl;
        return run_load_model_phase_one();
    }

    Model model = models[index - 1];

    std::cout << "Enter the file name of the model you want to load:" << std::endl;

    std::string file_name;

    std::cin >> file_name;

  ml::Sequential net = model.create_network();
    if (!net.load(file_name)) {
        std::cout << "Error: Model file could not be loaded. Please check the path and try again." << std::endl;
        return;
    }

    std::cout << "Enter the board dimension you want to play against the model on: ";

    int board_dimension;

    std::cin >> board_dimension;

    PhaseTwoOption phase_two_option = get_phase_two_option();

    if (phase_two_option == PhaseTwoOption::PLAY_AGAINST_MODEL) {
        play_against_model(net, board_dimension);
    }

    else if (phase_two_option == PhaseTwoOption::PLAY_AGAINST_RANDOM) {
        play_many_random_games(net, board_dimension, 1000);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    
}

int main() {

    PhaseOneOption phase_one_option = get_phase_one_option();

    if (phase_one_option == PhaseOneOption::RUN_EXPERIMENT) {
        run_experiment_phase_one();
    }

    else if (phase_one_option == PhaseOneOption::LOAD_MODEL) {
        run_load_model_phase_one();
    }
    
    return 0;
}
