# Root driver for C-torch (CMake is canonical). Examples:
#   make build          # configure + compile default targets
#   make test           # build + ctest
#   make py FILE=foo.py ARGS="..." # run a Python script with PYTHONPATH=./python (ctorch bindings)
#   make classification # only the Iris experiment (Make in subdir)
#
# Recommender via CMake (needs libcurl, Python dev headers, NumPy):
#   cmake -B build -DCTORCH_BUILD_RECOMMENDER=ON
#   cmake --build build --target recommender

BUILD_DIR ?= build
CMAKE ?= cmake
CMAKE_FLAGS ?= -DCMAKE_BUILD_TYPE=Release
PYTHON ?= python3

.PHONY: help configure build test clean classification ndtictactoe kernel-demo linear-regression-demo logistic-regression-demo distributed-logistic-demo distributed-classical-demo recommender-cmake py py-bindings py-wheel

help:
	@echo "Targets:"
	@echo "  make configure   - $(CMAKE) -B $(BUILD_DIR) $(CMAKE_FLAGS) ."
	@echo "  make build       - configure then compile (classification, tictactoe, ttt_main, tests)"
	@echo "  make test        - build then ctest --output-on-failure"
	@echo "  make py FILE=... [ARGS=...] - python3 FILE with PYTHONPATH=$(CURDIR)/python (ctorch bindings)"
	@echo "  make py-bindings - build ctorch_c shared library (required by Python package)"
	@echo "  make py-wheel    - build wheel in ./dist after staging native ctorch_c library"
	@echo "  make kernel-demo - build the concentric-circles Kernel SVM demo"
	@echo "  make linear-regression-demo - build the noisy line Linear Regression demo"
	@echo "  make logistic-regression-demo - build the two-blob Logistic Regression demo"
	@echo "  make distributed-logistic-demo - build the distributed logistic regression demo"
	@echo "  make distributed-classical-demo - build the distributed classical-model demo"
	@echo "  make classification / ndtictactoe - build experiments with their Makefiles"
	@echo "  make recommender-cmake - print CMake line to build recommender (optional deps)"
	@echo "  make clean       - rm -rf $(BUILD_DIR)"

py:
	@test -n "$(FILE)" || (echo "Usage: make py FILE=your_script.py" >&2 && false)
	PYTHONPATH="$(CURDIR)/python" $(PYTHON) $(FILE) $(ARGS)

py-bindings:
	$(CMAKE) -B $(BUILD_DIR) -DCTORCH_BUILD_TESTS=OFF -DCTORCH_BUILD_PYTHON_BINDINGS=ON $(CMAKE_FLAGS) .
	$(CMAKE) --build $(BUILD_DIR) --target ctorch_c --parallel

py-wheel:
	BUILD_DIR="$(BUILD_DIR)" PYTHON_BIN="$(PYTHON)" bash scripts/build_python_wheel.sh

configure:
	$(CMAKE) -B $(BUILD_DIR) $(CMAKE_FLAGS) .

build: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel

test: build
	$(CMAKE) -E chdir $(BUILD_DIR) ctest --output-on-failure

clean:
	rm -rf $(BUILD_DIR)

classification:
	$(MAKE) -C experiments/classification

ndtictactoe:
	$(MAKE) -C experiments/ndtictactoe

kernel-demo: configure
	$(CMAKE) --build $(BUILD_DIR) --target kernel_circle_demo --parallel

linear-regression-demo: configure
	$(CMAKE) --build $(BUILD_DIR) --target linear_regression_demo --parallel

logistic-regression-demo: configure
	$(CMAKE) --build $(BUILD_DIR) --target logistic_regression_demo --parallel

distributed-logistic-demo: configure
	$(CMAKE) --build $(BUILD_DIR) --target distributed_logistic_demo --parallel

distributed-classical-demo: configure
	$(CMAKE) --build $(BUILD_DIR) --target distributed_classical_models --parallel

recommender-cmake:
	@echo "$(CMAKE) -B $(BUILD_DIR) -DCTORCH_BUILD_RECOMMENDER=ON $(CMAKE_FLAGS) . && $(CMAKE) --build $(BUILD_DIR) --target recommender"
