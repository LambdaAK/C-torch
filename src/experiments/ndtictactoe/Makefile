# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -std=c++17 -I../../math -I../../ml -I../../rl -march=native -funroll-loops

# Source files
MAIN_SRC = ./main.cpp
MATH_SRC = ../../math/matrix.cpp
NN_SRC = ../../ml/nn.cpp
SAMPLE_SRC = ./sample.cpp
TICTACTOE_SRC = ./tictactoe.cpp
REPLAY_SRC = ./replaymemory.cpp
DQN_SRC = ./dqn.cpp
REINFORCE_SRC = ./reinforce.cpp
UCB_SRC = ../../ml/ucb.cpp

# Output binary name
TARGET = tictactoe

all: $(TARGET)

$(TARGET): $(MAIN_SRC) $(MATH_SRC) $(NN_SRC) $(SAMPLE_SRC) $(TICTACTOE_SRC) $(REPLAY_SRC) $(DQN_SRC) $(REINFORCE_SRC) $(UCB_SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(MAIN_SRC) $(MATH_SRC) $(NN_SRC) $(SAMPLE_SRC) $(TICTACTOE_SRC) $(REPLAY_SRC) $(DQN_SRC) $(REINFORCE_SRC) $(UCB_SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
