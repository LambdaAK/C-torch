#include "dqn.hpp"
#include "reinforce.hpp"
#include "tictactoe.hpp"
#include <chrono>
#include <thread>
#include <random>

#define LOG(x) std::cout << x << std::endl

/**
 * Configuration struct for the N-dimensional Tic Tac Toe DQN agent.
 */
struct Config
{
  size_t board_size;         // Size of one side of the square game board (e.g. 3 for 3x3, 4 for 4x4)
  std::string model_path;    // Path to save/load the trained model weights
  ml::Sequential q_net;      // Main Q-network for action selection and training
  ml::Sequential target_net; // Target network used to compute target Q-values during training
};

std::pair<ml::Sequential, ml::Sequential> create_networks(size_t board_size)
{
  size_t input_size = 2 * board_size * board_size; // state representation size
  size_t output_size = board_size * board_size;    // number of possible actions

  // Q-network creation
  ml::Sequential q_net;
  q_net.add_layer(std::make_shared<ml::LinearLayer>(input_size, 128));
  q_net.add_layer(std::make_shared<ml::ReLULayer>());
  q_net.add_layer(std::make_shared<ml::LinearLayer>(128, 128));
  q_net.add_layer(std::make_shared<ml::ReLULayer>());
  q_net.add_layer(std::make_shared<ml::LinearLayer>(128, output_size));

  // target network creation
  ml::Sequential target_net;
  target_net.add_layer(std::make_shared<ml::LinearLayer>(input_size, 128));
  target_net.add_layer(std::make_shared<ml::ReLULayer>());
  target_net.add_layer(std::make_shared<ml::LinearLayer>(128, 128));
  target_net.add_layer(std::make_shared<ml::ReLULayer>());
  target_net.add_layer(std::make_shared<ml::LinearLayer>(128, output_size));

  return {q_net, target_net};
}

void train(ml::Sequential q_net, ml::Sequential target_net, std::string &save_dir, size_t board_size, int episodes, int batch_size = 64)
{
  DQNAgent agent(q_net, target_net, 0.99f, 0.1f, 0.99995f, 0.9f, 0.001f, batch_size);

  TicTacToe env(board_size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 1);

  // 1 episode = 1 game
  for (int episode = 0; episode < episodes; ++episode)
  {
    env.reset();
    std::vector<float> state = env.get_state();
    bool done = false;

    bool agent_turn = true;

    while (!done)
    {
      std::vector<int> valid_actions = env.get_valid_actions();

      if (agent_turn)
      {
        int action = agent.act(state, valid_actions);
        auto [next_state, reward] = env.step(action);
        done = env.is_game_over();

        Sample s(state, action, reward, next_state, done);
        agent.add_to_memory(s);
        state = next_state;
      }
      else
      {
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

    if ((episode + 1) % 100 == 0)
    {
      std::cout << "Episode: " << episode + 1 << ", Epsilon: " << agent.get_epsilon() << std::endl;
    }

    if ((episode + 1) % 20000 == 0)
    {
      agent.save(save_dir + "dqn_tictactoe_" + std::to_string(episode + 1) + ".model");
    }
  }
}

void train_reinforce(ml::Sequential policy_net, ml::Sequential critic_net, std::string &path, size_t board_size, int episodes)
{
  // Initialize the REINFORCE agent with improved parameters
  float lr = 0.001f;             // Learning rate
  float gamma = 0.99f;           // Discount factor
  size_t batch_size = 32;        // Smaller batch size for more frequent updates
  bool norm_traj = true;         // Whether to normalize trajectories
  bool subtract_baseline = true; // Use advantage function
  bool rew_to_go = true;         // Whether to use rewards-to-go

  Reinforce agent(policy_net, critic_net, 0, lr, gamma, batch_size, norm_traj, subtract_baseline, rew_to_go);

  TicTacToe env(board_size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 1);

  int eval_interval = 1000;
  int eval_games = 100;
  float best_win_rate = 0.0f;

  int wins = 0, losses = 0, draws = 0;

  size_t batch_count = 0;
  for (int episode = 0; episode < episodes; ++episode)
  {
    env.reset();
    std::vector<float> state = env.get_state();
    bool done = false;
    bool agent_turn = true; 

    float episode_reward = 0.0f;

    while (!done)
    {
      std::vector<int> valid_actions = env.get_valid_actions();

      if (agent_turn)
      {
        int action = agent.act(batch_count, state, valid_actions);

        auto [next_state, reward] = env.step(action);
        episode_reward += reward;
        agent.update_last_reward(batch_count, reward);

        done = env.is_game_over();
        state = next_state;
      }
      else
      {
        std::uniform_int_distribution<> action_dis(0, valid_actions.size() - 1);
        int action = valid_actions[action_dis(gen)];
        auto [next_state, reward] = env.step(action);

        episode_reward -= reward;
        done = env.is_game_over();
        state = next_state;
      }

      agent_turn = !agent_turn;
    }

    int winner = env.get_winner();
    if (winner == 1)
      wins++;
    else if (winner == 2)
      losses++;
    else
      draws++;

    batch_count++;

    // update network on batch completion
    if (batch_count == batch_size)
    {
      agent.update_network();
      batch_count = 0;
    }

    // periodically show statistics
    if ((episode + 1) % 500 == 0)
    {
      float win_rate = static_cast<float>(wins) / 500;
      float loss_rate = static_cast<float>(losses) / 500;
      float draw_rate = static_cast<float>(draws) / 500;

      std::cout << "Episode: " << episode + 1
                << " | Win rate: " << win_rate
                << " | Loss rate: " << loss_rate
                << " | Draw rate: " << draw_rate << std::endl;

      wins = 0;
      losses = 0;
      draws = 0;
    }

    if ((episode + 1) % eval_interval == 0)
    {
      int eval_wins = 0, eval_losses = 0, eval_draws = 0;

      for (int eval_game = 0; eval_game < eval_games; eval_game++)
      {
        TicTacToe eval_env(board_size);
        eval_env.reset();
        std::vector<float> eval_state = eval_env.get_state();
        bool eval_done = false;
        bool eval_agent_turn = true;

        while (!eval_done)
        {
          std::vector<int> eval_valid_actions = eval_env.get_valid_actions();

          if (eval_agent_turn)
          {
            int eval_action = agent.act(0, eval_state, eval_valid_actions, true); // set inference mode
            auto [next_eval_state, _] = eval_env.step(eval_action);
            eval_state = next_eval_state;
          }
          else
          {
            std::uniform_int_distribution<> eval_action_dis(0, eval_valid_actions.size() - 1);
            int random_action = eval_valid_actions[eval_action_dis(gen)];
            auto [next_eval_state, _] = eval_env.step(random_action);
            eval_state = next_eval_state;
          }

          eval_done = eval_env.is_game_over();
          eval_agent_turn = !eval_agent_turn;
        }

        int eval_winner = eval_env.get_winner();
        if (eval_winner == 1)
          eval_wins++;
        else if (eval_winner == 2)
          eval_losses++;
        else
          eval_draws++;
      }

      float eval_win_rate = static_cast<float>(eval_wins) / eval_games;
      std::cout << "Evaluation after " << episode + 1 << " episodes: Win rate = "
                << eval_win_rate << ", Losses = " << eval_losses << ", Draws = " << eval_draws << std::endl;

      // save if this is the best model so far
      if (eval_win_rate > best_win_rate)
      {
        best_win_rate = eval_win_rate;
        agent.save(path + "reinforce_tictactoe_best.model");
        std::cout << "New best model saved with win rate: " << best_win_rate << std::endl;
      }

      // also save on regular checkpoints
      if ((episode + 1) % 20000 == 0)
      {
        agent.save(path + "reinforce_tictactoe_" + std::to_string(episode + 1) + ".model");
      }
    }
  }

  agent.save(path + "reinforce_tictactoe_final.model");
}

int play(const std::string &model_path, size_t board_size, bool random_human, ml::Sequential &q_net, ml::Sequential &target_net, bool no_delay = false)
{
  DQNAgent agent(q_net, target_net, 0.99f, 0.1f, 0.99995f, 0.9f, 0.001f, 64);
  agent.load(model_path);
  agent.set_epsilon(0.0f);

  TicTacToe env(board_size);
  env.reset();
  bool done = false;

  std::cout << "Playing against trained agent. You are O, agent is X." << std::endl;
  if (!random_human)
  {
    std::cout << "Board positions are numbered 0-" << (board_size * board_size - 1) << " as follows:" << std::endl;
    for (size_t i = 0; i < board_size; ++i)
    {
      for (size_t j = 0; j < board_size; ++j)
      {
        std::cout << std::setw(3) << (i * board_size + j) << " ";
      }
      std::cout << std::endl;
      if (i < board_size - 1)
      {
        std::cout << std::string(board_size * 4 - 1, '-') << std::endl;
      }
    }
    std::cout << std::endl;
  }

  while (!done)
  {
    env.render();

    int action;
    if (env.get_current_player() == 1)
    {
      std::vector<float> state = env.get_state();
      std::vector<int> valid_actions = env.get_valid_actions();
      action = agent.act(state, valid_actions);
      std::cout << "Agent plays position: " << action << std::endl;
    }
    else
    {
      std::vector<int> valid_actions = env.get_valid_actions();
      if (random_human)
      {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, valid_actions.size() - 1);
        action = valid_actions[dis(gen)];
        std::cout << "Random move selected: " << action << std::endl;
      }
      else
      {
        do
        {
          std::cout << "Enter your move (0-" << (board_size * board_size - 1) << "): ";
          std::cin >> action;
          if (action < 0 || action >= board_size * board_size)
          {
            std::cout << "Invalid move! Please enter a number between 0 and " << (board_size * board_size - 1) << "." << std::endl;
            continue;
          }
        } while (std::find(valid_actions.begin(), valid_actions.end(), action) == valid_actions.end());
      }
    }
    auto [next_state, reward] = env.step(action);
    done = env.is_game_over();
    if (random_human && !no_delay)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }
  env.render();
  int winner = env.get_winner();
  if (winner == 1)
  {
    std::cout << "Agent wins!" << std::endl;
    if (!no_delay)
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  }
  else if (winner == 2)
  {
    std::cout << "You win!" << std::endl;
    if (!no_delay)
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  }
  else
  {
    std::cout << "It's a draw!" << std::endl;
    if (!no_delay)
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  }
  return winner;
}

int play_reinforce(const std::string &model_path, size_t board_size, bool random_human, ml::Sequential &policy_network, bool no_delay = false)
{
  Reinforce agent(policy_network, policy_network);
  agent.load(model_path);

  TicTacToe env(board_size);
  env.reset();
  bool done = false;

  std::cout << "Playing against trained agent. You are O, agent is X." << std::endl;
  if (!random_human)
  {
    std::cout << "Board positions are numbered 0-" << (board_size * board_size - 1) << " as follows:" << std::endl;
    for (size_t i = 0; i < board_size; ++i)
    {
      for (size_t j = 0; j < board_size; ++j)
      {
        std::cout << std::setw(3) << (i * board_size + j) << " ";
      }
      std::cout << std::endl;
      if (i < board_size - 1)
      {
        std::cout << std::string(board_size * 4 - 1, '-') << std::endl;
      }
    }
    std::cout << std::endl;
  }

  while (!done)
  {
    env.render();

    int action;
    if (env.get_current_player() == 1)
    {
      std::vector<float> state = env.get_state();
      std::vector<int> valid_actions = env.get_valid_actions();
      std::cout << "REACH" << std::endl;
      action = agent.act(0, state, valid_actions, true, true);
      std::cout << "Agent plays position: " << action << std::endl;
    }
    else
    {
      std::vector<int> valid_actions = env.get_valid_actions();
      if (random_human)
      {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, valid_actions.size() - 1);
        action = valid_actions[dis(gen)];
        std::cout << "Random move selected: " << action << std::endl;
      }
      else
      {
        do
        {
          std::cout << "Enter your move (0-" << (board_size * board_size - 1) << "): ";
          std::cin >> action;
          if (action < 0 || action >= board_size * board_size)
          {
            std::cout << "Invalid move! Please enter a number between 0 and " << (board_size * board_size - 1) << "." << std::endl;
            continue;
          }
        } while (std::find(valid_actions.begin(), valid_actions.end(), action) == valid_actions.end());
      }
    }
    auto [next_state, reward] = env.step(action);
    done = env.is_game_over();
    if (random_human && !no_delay)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }
  env.render();
  int winner = env.get_winner();
  if (winner == 1)
  {
    std::cout << "Agent wins!" << std::endl;
    if (!no_delay)
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  }
  else if (winner == 2)
  {
    std::cout << "You win!" << std::endl;
    if (!no_delay)
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  }
  else
  {
    std::cout << "It's a draw!" << std::endl;
    if (!no_delay)
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  }
  return winner;
}

Config make_config1(std::string model_path)
{

  std::cout << "aaaa" << std::endl;

  size_t board_size = 3;

  ml::Sequential q_net;
  q_net.add_layer(std::make_shared<ml::LinearLayer>(18, 32));
  q_net.add_layer(std::make_shared<ml::ReLULayer>());
  q_net.add_layer(std::make_shared<ml::LinearLayer>(32, 32));
  q_net.add_layer(std::make_shared<ml::ReLULayer>());
  q_net.add_layer(std::make_shared<ml::LinearLayer>(32, 9));

  ml::Sequential target_net;
  target_net.add_layer(std::make_shared<ml::LinearLayer>(18, 32));
  target_net.add_layer(std::make_shared<ml::ReLULayer>());
  target_net.add_layer(std::make_shared<ml::LinearLayer>(32, 32));
  target_net.add_layer(std::make_shared<ml::ReLULayer>());
  target_net.add_layer(std::make_shared<ml::LinearLayer>(32, 9));

  Config config;
  config.board_size = board_size;
  config.model_path = model_path;
  config.q_net = q_net;
  config.target_net = target_net;
  return config;
}

Config make_config2(std::string model_path)
{

  size_t board_size = 3;

  std::shared_ptr<ml::LinearLayer> input = std::make_shared<ml::LinearLayer>(18, 64);
  std::shared_ptr<ml::ReLULayer> r1 = std::make_shared<ml::ReLULayer>();
  std::shared_ptr<ml::LinearLayer> h1 = std::make_shared<ml::LinearLayer>(64, 64);
  std::shared_ptr<ml::ReLULayer> r2 = std::make_shared<ml::ReLULayer>();
  std::shared_ptr<ml::LinearLayer> output = std::make_shared<ml::LinearLayer>(64, 9);

  ml::Sequential q_net;
  q_net.add_layer(input);
  q_net.add_layer(r1);
  q_net.add_layer(h1);
  q_net.add_layer(r2);
  q_net.add_layer(output);

  input = std::make_shared<ml::LinearLayer>(18, 128);
  r1 = std::make_shared<ml::ReLULayer>();
  h1 = std::make_shared<ml::LinearLayer>(128, 64);
  r2 = std::make_shared<ml::ReLULayer>();
  output = std::make_shared<ml::LinearLayer>(64, 1);

  ml::Sequential target_net;
  target_net.add_layer(input);
  target_net.add_layer(r1);
  target_net.add_layer(h1);
  target_net.add_layer(r2);
  target_net.add_layer(output);

  Config config;
  config.board_size = board_size;
  config.model_path = model_path;
  config.q_net = q_net;
  config.target_net = target_net;
  return config;
}

Config make_config3(std::string model_path)
{
  size_t board_size = 4;

  // 32 -> 128 -> ReLU -> 128 -> ReLU -> 16

  std::shared_ptr<ml::LinearLayer> input = std::make_shared<ml::LinearLayer>(32, 128);
  std::shared_ptr<ml::ReLULayer> r1 = std::make_shared<ml::ReLULayer>();
  std::shared_ptr<ml::LinearLayer> h1 = std::make_shared<ml::LinearLayer>(128, 128);
  std::shared_ptr<ml::ReLULayer> r2 = std::make_shared<ml::ReLULayer>();
  std::shared_ptr<ml::LinearLayer> output = std::make_shared<ml::LinearLayer>(128, 16);

  ml::Sequential q_net;
  q_net.add_layer(input);
  q_net.add_layer(r1);
  q_net.add_layer(h1);
  q_net.add_layer(r2);
  q_net.add_layer(output);

  input = std::make_shared<ml::LinearLayer>(32, 128);
  r1 = std::make_shared<ml::ReLULayer>();
  h1 = std::make_shared<ml::LinearLayer>(128, 128);
  r2 = std::make_shared<ml::ReLULayer>();
  output = std::make_shared<ml::LinearLayer>(128, 16);

  ml::Sequential target_net;
  target_net.add_layer(input);
  target_net.add_layer(r1);
  target_net.add_layer(h1);
  target_net.add_layer(r2);
  target_net.add_layer(output);

  Config config;
  config.board_size = board_size;
  config.model_path = model_path;
  config.q_net = q_net;
  config.target_net = target_net;
  return config;
}

int main()
{
  std::string path;
  std::cout << "Enter path to model or training output directory\n";

  std::getline(std::cin, path);
  if (!path.empty() && path.front() == '"' && path.back() == '"')
  {
    path = path.substr(1, path.size() - 2);
  }
  namespace fs = std::filesystem;
  if (fs::exists(path))
  {
    std::cout << "Model path exists!\n"
              << std::endl;
  }
  else
  {
    std::cout << "Model path does not exist.\n"
              << std::endl;
    return 1;
  }

  std::cout
      << "Available configurations:\n";
  std::cout << "1. 3x3 Board with 32 hidden units\n";
  std::cout << "2. 3x3 Board with 128 hidden units\n";
  std::cout << "3. 4x4 Board with 128 hidden units\n";
  std::cout << "Enter configuration choice (1-3): ";

  int config_choice;
  std::cin >> config_choice;

  Config config;
  switch (config_choice)
  {
  case 1:
    config = make_config1(path);
    break;
  case 2:
    config = make_config2(path);
    break;
  case 3:
    config = make_config3(path);
    break;
  default:
    std::cout << "Invalid configuration choice!" << std::endl;
    return 1;
  }

  std::cout << "\nSelected configuration:\n";
  std::cout << "Board size: " << config.board_size << "x" << config.board_size << "\n";
  std::cout << "Model path: " << config.model_path << "\n\n";

  std::cout
      << "Input Model Type:\n";
  std::cout << "1: DQN Agent\n";
  std::cout << "2. REINFORCE Agent\n";
  std::cout << "3. Actor/Critic REINFORCE Agent\n";
  std::cout << "4. Double DQN Agent\n\n";

  int choice;
  std::cin >> choice;

  std::cout << "Select mode:\n";
  std::cout << "1. AI vs Human\n";
  std::cout << "2. AI vs Random\n";
  std::cout << "3. Train new model\n";
  std::cout << "4. Quick Random\n";
  std::cout << "Enter choice (1-5): ";

  int mode_choice;
  std::cin >> mode_choice;

  int agent_wins = 0;
  int opponent_wins = 0;
  int draws = 0;
  const int num_games = 1000;

  if (mode_choice == 1)
  {
    // AI vs Human
    for (size_t i = 0; i < num_games; ++i)
    {
      int winner;
      if (choice == 1)
      {
        LOG("REACH");
        winner = play(config.model_path, config.board_size, false, config.q_net, config.target_net);
      }
      else if (choice == 2)
      {
        LOG("REACH");
        winner = play_reinforce(config.model_path, config.board_size, false, config.q_net);
      }
      if (winner == 1)
        agent_wins++;
      else if (winner == 2)
        opponent_wins++;
      else
        draws++;
      std::cout << "\nCurrent Statistics:" << std::endl;
      std::cout << "Agent wins: " << agent_wins << std::endl;
      std::cout << "Human wins: " << opponent_wins << std::endl;
      std::cout << "Draws: " << draws << std::endl;
      std::cout << "Total games: " << (agent_wins + opponent_wins + draws) << std::endl;
    }
  }
  else if (mode_choice == 2)
  {
    // AI vs Random
    for (size_t i = 0; i < num_games; ++i)
    {
      int winner;
      if (choice == 1)
      {
        winner = play(config.model_path, config.board_size, true, config.q_net, config.target_net);
      }
      else if (choice == 2)
      {
        winner = play_reinforce(config.model_path, config.board_size, true, config.q_net);
      }
      if (winner == 1)
        agent_wins++;
      else if (winner == 2)
        opponent_wins++;
      else
        draws++;
      std::cout << "\nCurrent Statistics:" << std::endl;
      std::cout << "Agent wins: " << agent_wins << std::endl;
      std::cout << "Random player wins: " << opponent_wins << std::endl;
      std::cout << "Draws: " << draws << std::endl;
      std::cout << "Total games: " << (agent_wins + opponent_wins + draws) << std::endl;
    }
  }
  else if (mode_choice == 3)
  {
    std::cout << "selected mode 3" << std::endl;
    // Train new model
    if (choice == 1)
    {
      train(config.q_net, config.target_net, path, config.board_size, 200000, 64);
    }
    else if (choice == 2)
    {
      train_reinforce(config.q_net, config.target_net, path, config.board_size, 500000);
    }
  }
  else if (mode_choice == 4)
  {
    const int quick_random_games = 10000;
    for (size_t i = 0; i < quick_random_games; ++i)
    {
      int winner;
      if (choice == 1)
      {
        winner = play(config.model_path, config.board_size, true, config.q_net, config.target_net, true);
      }
      else if (choice == 2)
      {
        winner = play_reinforce(config.model_path, config.board_size, true, config.q_net, true);
      }
      if (winner == 1)
        agent_wins++;
      else if (winner == 2)
        opponent_wins++;
      else
        draws++;
      std::cout << "\nCurrent Statistics:" << std::endl;
      std::cout << "Agent wins: " << agent_wins << std::endl;
      std::cout << "Random player wins: " << opponent_wins << std::endl;
      std::cout << "Draws: " << draws << std::endl;
      std::cout << "Total games: " << (agent_wins + opponent_wins + draws) << std::endl;
    }
  }
}
