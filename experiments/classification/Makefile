# Classification experiment — also buildable via CMake from repo root:
#   cmake -B build && cmake --build build --target classification
# Example distributed run:
#   make run ARGS="--distributed --rank 0 --world-size 2 --master-port 29500"

CXX = g++
CXXFLAGS = -std=c++20 -O3 -Wall -Wextra -pthread -I../../lib -I../recommender
ARGS ?=

SRC_DIR = .
MATH_DIR = ../../lib/math
ML_DIR = ../../lib/ml
DIST_DIR = ../../lib/distributed

SOURCES = \
	$(SRC_DIR)/main.cpp \
	$(MATH_DIR)/matrix.cpp \
	$(MATH_DIR)/dataaugmentor.cpp \
	$(ML_DIR)/perceptron.cpp \
	$(ML_DIR)/svm.cpp \
	$(ML_DIR)/linearregression.cpp \
	$(ML_DIR)/logisticregression.cpp \
	$(ML_DIR)/knn.cpp \
	$(ML_DIR)/nn.cpp \
	$(ML_DIR)/nn_optim.cpp \
	$(ML_DIR)/randomfouriersvm.cpp \
	$(ML_DIR)/kernelsvm.cpp \
	$(DIST_DIR)/tcp_process_group.cpp \
	$(ML_DIR)/gaussian_nb.cpp

TARGET = main

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCES)

run: $(TARGET)
	./$(TARGET) $(ARGS)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
