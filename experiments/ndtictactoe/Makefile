# N-D tic-tac-toe — also buildable via CMake from repo root:
#   cmake -B build && cmake --build build --target tictactoe ttt_main

CXX = g++
CXXFLAGS = -Wall -std=c++17 -pthread -I../../lib -march=native -funroll-loops

MAIN_SRC = ./main.cpp
MATH_SRC = ../../lib/math/matrix.cpp
NN_SRC = ../../lib/ml/nn.cpp
NN_OPT_SRC = ../../lib/ml/nn_optim.cpp
SAMPLE_SRC = ./sample.cpp
TICTACTOE_SRC = ./tictactoe.cpp
REPLAY_SRC = ./replaymemory.cpp
DQN_SRC = ./dqn.cpp
REINFORCE_SRC = ./reinforce.cpp
UCB_SRC = ../../lib/ml/ucb.cpp

TARGET = tictactoe

all: $(TARGET)

$(TARGET): $(MAIN_SRC) $(MATH_SRC) $(NN_SRC) $(NN_OPT_SRC) $(SAMPLE_SRC) $(TICTACTOE_SRC) $(REPLAY_SRC) $(DQN_SRC) $(REINFORCE_SRC) $(UCB_SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(MAIN_SRC) $(MATH_SRC) $(NN_SRC) $(NN_OPT_SRC) $(SAMPLE_SRC) $(TICTACTOE_SRC) $(REPLAY_SRC) $(DQN_SRC) $(REINFORCE_SRC) $(UCB_SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
