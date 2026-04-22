# Standalone build: vendor nlohmann/json under json/single_include/ or use system <nlohmann/json.hpp>.
# Prefer the CMake target with -DCTORCH_BUILD_RECOMMENDER=ON (fetches json automatically).
CXX = g++
CXXFLAGS = -std=c++20 -O3 -pthread -lcurl -I../../lib -Ijson/single_include

# Python configuration
PYTHON_VERSION = 3.11
PYTHON_INCLUDE = $(shell python$(PYTHON_VERSION)-config --includes)
NUMPY_INCLUDE = $(shell python3 -c "import numpy; print(numpy.get_include())")
PYTHON_LIBDIR = $(shell python$(PYTHON_VERSION)-config --prefix)/lib
PYTHON_LIBS = -lpython$(PYTHON_VERSION)

# Source files
SRC = main.cpp \
      ../../lib/math/matrix.cpp \
      ../../lib/ml/kmeans.cpp \
      ../../lib/ml/mab.cpp \
      ../../lib/ml/pca.cpp \
      ../../lib/ml/ucb.cpp

# Output
OUT = main

# Build target
all:
	$(CXX) $(CXXFLAGS) $(SRC) $(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) -L$(PYTHON_LIBDIR) $(PYTHON_LIBS) -o $(OUT)

# Clean target
clean:
	rm -f $(OUT)
