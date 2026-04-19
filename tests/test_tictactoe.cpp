#include <gtest/gtest.h>
#include "tictactoe.hpp"

TEST(TicTacToe, RejectsZeroBoardSize) {
    EXPECT_THROW(TicTacToe game(0), std::invalid_argument);
}

TEST(TicTacToe, SetPlayerToOptimizeForValidatesInput) {
    TicTacToe game(3);
    EXPECT_NO_THROW(game.set_player_to_optimize_for(1));
    EXPECT_NO_THROW(game.set_player_to_optimize_for(2));
    EXPECT_THROW(game.set_player_to_optimize_for(0), std::invalid_argument);
    EXPECT_THROW(game.set_player_to_optimize_for(3), std::invalid_argument);
}
