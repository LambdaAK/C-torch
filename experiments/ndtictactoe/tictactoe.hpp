#pragma once

#include <cstddef>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <memory>
#include <deque>
#include <cmath>

class TicTacToe
{
private:
  size_t board_size;                   // size of the board (n x n)
  std::vector<std::vector<int>> board; // each position on the board
  int cur_player;                      // current player (1 or 2)
  bool game_over;                      // whether the game is over
  int winner;                          // winner of the game (0 if no winner, 1 if player 1 wins, 2 if player 2 wins)
  bool check_winner(int player);       // check if the current player has won
  bool check_draw();                   // check if the game is a draw
  int player_to_optimize_for;          // 1 = player 1, 2 = player 2

  // New helper functions for reward calculation
  float evaluate_move_quality(int row, int col, int player);
  float evaluate_potential_wins(int row, int col, int player);
  int count_empty_cells() const;

public:
  /**
   * Creates new Tic-tac-toe game
   * @param size The size of the board (n x n)
   */
  TicTacToe(size_t size = 4);

  /**
   * Returns the size of the board
   */
  size_t get_board_size() const;

  /**
   * Resets game to initial state
   */
  void reset();

  /**
   * Makes a move and returns new state and reward
   */
  std::pair<std::vector<float>, float> step(int action);

  /**
   * Returns current board state as vector
   */
  std::vector<float> get_state() const;

  /**
   * Returns list of valid moves
   */
  std::vector<int> get_valid_actions() const;

  /**
   * Returns if game has ended
   */
  bool is_game_over() const;

  /**
   * Returns winner (0=none, 1=p1, 2=p2)
   */
  int get_winner() const;

  /**
   * Returns current player (1 or 2)
   */
  int get_current_player() const;

  /**
   * Prints board to console
   */
  void render() const;

  /**
   * Sets the player to optimize the reward function for
   * @param player 1 for player 1, 2 for player 2
   */
  void set_player_to_optimize_for(int player);
};