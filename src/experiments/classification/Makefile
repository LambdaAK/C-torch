# Makefile for src/experiments/classification/main.cpp

CXX = g++
CXXFLAGS = -std=c++17 -O3 -I../../ -I../../../

SRC_DIR = .
MATH_DIR = ../../math
ML_DIR = ../../ml

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
	$(ML_DIR)/kernelsvm.cpp

TARGET = main

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCES)

clean:
	rm -f $(TARGET)