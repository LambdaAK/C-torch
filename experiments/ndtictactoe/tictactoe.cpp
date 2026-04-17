#include "tictactoe.hpp"

TicTacToe::TicTacToe(size_t size) : board_size(size), board(size, std::vector<int>(size, 0)), cur_player(1), game_over(false), winner(0)
{
}

size_t TicTacToe::get_board_size() const
{
  return board_size;
}

void TicTacToe::reset()
{
  board = std::vector<std::vector<int>>(board_size, std::vector<int>(board_size, 0));
  cur_player = 1;
  game_over = false;
  winner = 0;
}

bool TicTacToe::check_winner(int player)
{
  for (size_t i = 0; i < board_size; ++i)
  {
    bool row_win = true;
    for (size_t j = 0; j < board_size; ++j)
    {
      if (board[i][j] != player)
      {
        row_win = false;
        break;
      }
    }
    if (row_win)
      return true;
  }

  for (size_t j = 0; j < board_size; ++j)
  {
    bool col_win = true;
    for (size_t i = 0; i < board_size; ++i)
    {
      if (board[i][j] != player)
      {
        col_win = false;
        break;
      }
    }
    if (col_win)
      return true;
  }

  bool diag1_win = true;
  for (size_t i = 0; i < board_size; ++i)
  {
    if (board[i][i] != player)
    {
      diag1_win = false;
      break;
    }
  }
  if (diag1_win)
  {
    return true;
  }

  bool diag2_win = true;
  for (size_t i = 0; i < board_size; ++i)
  {
    if (board[i][board_size - 1 - i] != player)
    {
      diag2_win = false;
      break;
    }
  }
  if (diag2_win)
    return true;

  return false;
}

bool TicTacToe::check_draw()
{
  for (size_t i = 0; i < board_size; ++i)
  {
    if (std::find(board[i].begin(), board[i].end(), 0) != board[i].end())
    {
      return false;
    }
  }
  return true;
}

std::pair<std::vector<float>, float> TicTacToe::step(int action)
{
  float reward = 0.0f;
  size_t total_cells = board_size * board_size;

  if (action < 0 || action >= total_cells || board[std::floor(action / board_size)][action % board_size] != 0 || game_over)
  {
    reward = -10.0f;
    return {get_state(), reward};
  }

  board[std::floor(action / board_size)][action % board_size] = cur_player;

  if (cur_player == 1)
  {
    board[std::floor(action / board_size)][action % board_size] = 2;
    if (check_winner(2))
    {
      reward += 0.6; 
    }
    board[std::floor(action / board_size)][action % board_size] = cur_player;
  }

  if (check_winner(cur_player))
  {
    game_over = true;
    winner = cur_player;
    reward = cur_player == 1 ? 1.0f : -1.0f;
  }
  else if (check_draw())
  {
    game_over = true;
    reward = 0.1;
  }

  cur_player = cur_player == 1 ? 2 : 1;

  return {get_state(), reward};
}

// Helper function to evaluate move quality
float TicTacToe::evaluate_move_quality(int row, int col, int player)
{
  float quality = 0.0f;

  int opponent = (player == 1) ? 2 : 1;

  board[row][col] = opponent;
  bool blocks_win = check_winner(opponent);
  board[row][col] = player;

  if (blocks_win)
  {
    quality += 3.0f;
  }

  if (board_size % 2 == 1)
  {
    int center = board_size / 2;
    if (row == center && col == center)
    {
      quality += 0.2f;
    }
  }

  if ((row == 0 || row == board_size - 1) && (col == 0 || col == board_size - 1))
  {
    quality += 0.15f;
  }

  quality += evaluate_potential_wins(row, col, player);

  return quality;
}

// Helper function to count potential winning lines
float TicTacToe::evaluate_potential_wins(int row, int col, int player)
{
  float potential = 0.0f;

  int player_count = 0;
  int empty_count = 0;
  for (int c = 0; c < board_size; c++)
  {
    if (board[row][c] == player)
      player_count++;
    else if (board[row][c] == 0)
      empty_count++;
  }

  if (player_count + empty_count == board_size)
  {
    potential += 0.05f * player_count / static_cast<float>(board_size);
  }
  player_count = 0;
  empty_count = 0;
  for (int r = 0; r < board_size; r++)
  {
    if (board[r][col] == player)
      player_count++;
    else if (board[r][col] == 0)
      empty_count++;
  }

  if (player_count + empty_count == board_size)
  {
    potential += 0.05f * player_count / static_cast<float>(board_size);
  }

  if (row == col)
  {
    player_count = 0;
    empty_count = 0;
    for (int i = 0; i < board_size; i++)
    {
      if (board[i][i] == player)
        player_count++;
      else if (board[i][i] == 0)
        empty_count++;
    }

    if (player_count + empty_count == board_size)
    {
      potential += 0.05f * player_count / static_cast<float>(board_size);
    }
  }

  if (row + col == board_size - 1)
  {
    player_count = 0;
    empty_count = 0;
    for (int i = 0; i < board_size; i++)
    {
      if (board[i][board_size - 1 - i] == player)
        player_count++;
      else if (board[i][board_size - 1 - i] == 0)
        empty_count++;
    }

    if (player_count + empty_count == board_size)
    {
      potential += 0.05f * player_count / static_cast<float>(board_size);
    }
  }

  return potential;
}

// Helper to count empty cells
int TicTacToe::count_empty_cells() const
{
  int count = 0;
  for (int r = 0; r < board_size; r++)
  {
    for (int c = 0; c < board_size; c++)
    {
      if (board[r][c] == 0)
        count++;
    }
  }
  return count;
}

// State is a vector of 2 * board_size^2 (first half for player 1, second half for player 2)
std::vector<float> TicTacToe::get_state() const
{
  size_t total_cells = board_size * board_size;
  std::vector<float> state(2 * total_cells, 0.0f);

  for (size_t i = 0; i < board_size; ++i)
  {
    for (size_t j = 0; j < board_size; ++j)
    {
      if (board[i][j] == 1)
      {
        state[(i * board_size) + j] = 1.0f;
      }
      else if (board[i][j] == 2)
      {
        state[(i * board_size) + j + total_cells] = 1.0f;
      }
    }
  }

  return state;
}

std::vector<int> TicTacToe::get_valid_actions() const
{
  std::vector<int> valid_actions;
  for (size_t i = 0; i < board_size; ++i)
  {
    for (size_t j = 0; j < board_size; ++j)
    {
      if (board[i][j] == 0)
      {
        valid_actions.push_back((i * board_size) + j);
      }
    }
  }
  return valid_actions;
}

bool TicTacToe::is_game_over() const
{
  return game_over;
}

int TicTacToe::get_winner() const
{
  return winner;
}

int TicTacToe::get_current_player() const
{
  return cur_player;
}

void TicTacToe::render() const
{
  std::string separator = std::string(board_size * 4 + 1, '-');
  std::cout << separator << std::endl;
  for (size_t i = 0; i < board_size; ++i)
  {
    std::cout << "| ";
    for (size_t j = 0; j < board_size; ++j)
    {
      char symbol = ' ';
      if (board[i][j] == 1)
        symbol = 'X';
      else if (board[i][j] == 2)
        symbol = 'O';
      std::cout << symbol << " | ";
    }
    std::cout << std::endl
              << separator << std::endl;
  }
}
