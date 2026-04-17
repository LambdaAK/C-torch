# Classification experiment — also buildable via CMake from repo root:
#   cmake -B build && cmake --build build --target classification

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -I../../lib -I../recommender

SRC_DIR = .
MATH_DIR = ../../lib/math
ML_DIR = ../../lib/ml

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
	$(ML_DIR)/randomfouriersvm.cpp \
	$(ML_DIR)/kernelsvm.cpp \
	$(ML_DIR)/gaussian_nb.cpp

TARGET = main

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCES)

clean:
	rm -f $(TARGET)

.PHONY: all clean
