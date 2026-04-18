# C-torch

A from-scratch C++ machine learning project.  
The repository contains:

- A lightweight math layer (`Matrix`, symbolic AST expressions, numerical optimization helpers)
- ML algorithms implemented directly in C++
- End-to-end experiments for classification, recommendation, and reinforcement learning

This codebase is primarily experiment-driven and educational/research oriented.

## Quick reference

### Run a Python file with `ctorch` bindings

Build the `ctorch_c` shared library first (see [optional Python bindings](#optional-python-bindings-ctypes)). From the repository root:

```bash
make py FILE=your_script.py
```

Equivalent:

```bash
PYTHONPATH="$PWD/python" python3 your_script.py
```

## Table of contents

1. [Quick reference](#quick-reference)
2. [Repository layout](#repository-layout)
3. [Implemented components](#implemented-components)
4. [Requirements](#requirements)
5. [Build and run](#build-and-run)
6. [Experiments](#experiments)
7. [Data and artifacts](#data-and-artifacts)
8. [Known limitations](#known-limitations)
9. [Contributing](#contributing)
10. [Kernel SVM and Random Fourier notes](#kernel-svm-and-random-fourier-notes)
11. [Future work](#future-work)
12. [Project proposal](#project-proposal)

## Repository layout

```text
.
├── Makefile            # convenience: make build / make test / make py FILE=...
├── CMakeLists.txt      # unified build (classification + ndtictactoe + optional recommender)
├── scripts/            # e.g. git-untrack-artifacts.sh
├── lib/
│   ├── math/           # matrix ops, AST, differentiator, optimizers, augmentation
│   └── ml/             # ML models and utilities
├── experiments/
│   ├── classification/
│   ├── recommender/
│   └── ndtictactoe/
└── proposal/
    ├── Project Proposal.md
    └── Project Proposal.pdf
```

## Implemented components

### `lib/math`

- `matrix.*`
  - Dense matrix class
  - Element access, transpose
  - Arithmetic (`+`, `-`, scalar `*`, matrix `*`)
  - Vector distance and inner product
  - Activation helpers (`relu`, `sigmoid`, `tanh`) and derivatives
  - Column utilities (`l2_norm_cols`, `center_cols`)
- `ast.hpp`
  - Symbolic expression tree (`Num`, `Var`, arithmetic ops, `exp`, `log`, `sqrt`, `abs`, `max`, `min`, `sigmoid` helper)
  - Simplification and substitution
- `optim.hpp`
  - Numerical differentiation
  - Gradient Descent and SGD wrappers over AST-defined losses
- `dataaugmentor.*`
  - Feature expansion (`poly_2` to `poly_5`)
  - Random Fourier feature projection

### `lib/ml`

- Supervised:
  - Perceptron (`perceptron.*`)
  - Linear SVM (`svm.*`)
  - Kernel SVM (`kernelsvm.*`) — RBF kernel `exp(-gamma * ||x-y||^2)` (see [notes](#kernel-svm-and-random-fourier-notes))
  - Random Fourier SVM (`randomfouriersvm.*`) — cosine features with `gamma`-linked random frequencies (same section)
  - Logistic Regression (`logisticregression.*`)
  - Linear Regression (`linearregression.*`)
  - KNN (`knn.*`)
  - Gaussian Naive Bayes (`gaussian_nb.*`)
- Unsupervised / DR:
  - KMeans (`kmeans.*`)
  - PCA via QR iteration (`pca.*`)
- Reinforcement / bandits:
  - Epsilon-greedy MAB (`mab.*`)
  - UCB (`ucb.*`)
- Neural network primitives:
  - `Sequential`, `LinearLayer`, `ReLULayer`, `SigmoidLayer`, `TanhLayer`
  - SGD optimizer and simple model save/load (`nn.*`)

## Requirements

### Core

- C++17+ compiler (`g++` or `clang++`; CMake targets use C++20)
- `make` (per-experiment builds) or CMake 3.16+ (unified build from the repo root)

### Recommender experiment extras

- `libcurl`
- Python 3.11 development headers/libs
- NumPy include path available to Python
- `matplotlibcpp.h` is vendored. **CMake** recommender build fetches **nlohmann/json** via `FetchContent`. The **recommender `Makefile`** still expects a local `json/single_include/nlohmann/json.hpp` tree unless you build with CMake only.

### Notes

- The project is Linux/macOS oriented (system calls such as `open`/`xdg-open` are used in places).
- Paths are mostly relative to each experiment directory.

## Build and run

### CMake (recommended)

From the repository root:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

This produces `build/classification`, `build/tictactoe`, and `build/ttt_main`.  
Include paths use `-I lib` so sources include headers as `"math/..."` and `"ml/..."`.

Unit tests (GoogleTest, fetched on first configure) build with the default `CTORCH_BUILD_TESTS=ON`:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

To skip tests (no network fetch for googletest): `cmake -B build -DCTORCH_BUILD_TESTS=OFF`.

### Root `Makefile` (shortcuts)

```bash
make build    # cmake -B build && cmake --build build
make test     # build + ctest
make py FILE=your_script.py   # python3 with PYTHONPATH=./python (ctorch bindings)
make classification   # make -C experiments/classification
make ndtictactoe     # make -C experiments/ndtictactoe
```

### Optional: recommender (CMake)

Requires **libcurl**, **Python 3.9+** with **development headers**, and **NumPy** importable from that interpreter:

```bash
cmake -B build -DCTORCH_BUILD_RECOMMENDER=ON -DCMAKE_BUILD_TYPE=Release .
cmake --build build --target recommender
```

If configure fails, read the message: missing `CURL`, `Python3`, or `numpy` is the usual cause.

### Optional: Python bindings (ctypes)

This repository includes Python bindings for:

- `Matrix` (`shape`, element access, `+`, `-`, scalar `*`, matrix `@`, transpose)
- Supervised models:
  `KNN`, `LinearRegression`, `LogisticRegression`, `Perceptron`, `SVM`, `KernelSVM`, `RandomFourierSVM`, `GaussianNB`
- Unsupervised / bandits:
  `KMeans`, `PCA`, `MAB`, `UCB`
- Enum helpers:
  `OptimType`, `DataAugmentationType`, `KernelType`

Build the binding library:

```bash
cmake -B build -DCTORCH_BUILD_TESTS=OFF -DCTORCH_BUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ctorch_c --parallel
```

Run from the repo root (see [Quick reference](#quick-reference) for `make py FILE=...`).

Inline snippet:

```bash
PYTHONPATH="$PWD/python" python3 - <<'PY'
from ctorch import (
    Matrix,
    KNN,
    SVM,
    GaussianNB,
    KMeans,
    PCA,
    UCB,
    DataAugmentationType,
)

a = Matrix(data=[[1.0, 2.0], [3.0, 4.0]])
b = Matrix(data=[[5.0, 6.0], [7.0, 8.0]])
print((a + b).to_list())

x_tr = Matrix(data=[[0.0, 0.0], [10.0, 10.0], [1.0, 1.0]])
y_tr = Matrix(data=[[0.0, 1.0, 1.0]])
knn = KNN(1, x_tr, y_tr)
print(knn.predict([[0.1, 0.1]]))

svm = SVM(x_tr, [[-1.0, 1.0, 1.0]], learning_rate=0.0, max_iter=100, c_value=1.0, augmentation=DataAugmentationType.NO_OP)
print(svm.predict([[0.1, 0.1]]))

gnb = GaussianNB(x_tr, y_tr)
print(gnb.predict([[0.1, 0.1]]))

kmeans = KMeans(2, x_tr, max_iter=20)
print(kmeans.assignments)

pca = PCA([[-0.5, -0.5], [0.5, 0.5], [0.6, 0.4], [-0.6, -0.4]])
print(pca.compute_projection(1).shape)

ucb = UCB(3)
arm = ucb.select_arm()
ucb.update(arm, 1.0)
PY
```

If your shared library is not in `build/`, set `CTORCH_LIB_PATH` to the built `libctorch_c` path.

### Make (per experiment)

Each experiment directory still has a `Makefile` for direct `g++` builds.

### 1) Classification (Iris)

From `experiments/classification`, `Iris.csv` resolves by default. From the **repo root** (CMake `build/classification`), set **`CTORCH_IRIS_CSV`** to the CSV path (CI does this automatically).

```bash
cd experiments/classification
make
./main
```

```bash
# from repository root, after CMake build
CTORCH_IRIS_CSV="$PWD/experiments/classification/Iris.csv" ./build/classification
```

### 2) Recommender (KMeans + PCA + MAB/UCB)

```bash
cd experiments/recommender
make
./main
```

### 3) N-dimensional Tic-Tac-Toe RL (DQN/REINFORCE)

```bash
cd experiments/ndtictactoe
make
./tictactoe
```

There is also an alternate RL driver:

```bash
cd experiments/ndtictactoe
g++ -std=c++20 -O3 -I../../lib ttt_main.cpp tictactoe.cpp replaymemory.cpp dqn.cpp sample.cpp ../../lib/math/matrix.cpp ../../lib/ml/nn.cpp -o ttt_main
./ttt_main
```

## Experiments

### Classification (`experiments/classification`)

- Loads `Iris.csv` (or the path in **`CTORCH_IRIS_CSV`** if set)
- Performs train/test split and feature normalization
- Runs:
  - Linear SVM
  - Perceptron
  - Gaussian Naive Bayes (multiclass Iris)
  - KNN
  - Linear regression scoring variant
  - Feedforward NN classifier

### Recommender (`experiments/recommender`)

- Loads `processed_data.csv`
- Uses selected audio features to:
  - Optionally reduce dimension with PCA
  - Cluster tracks with KMeans
  - Treat clusters as arms for MAB/UCB
- Simulated reward is based on cosine similarity to a user preference vector
- Writes experiment metrics to JSON result files
- Includes optional Spotify track/image lookup utilities (requires auth token handling in code)

### N-D Tic-Tac-Toe RL (`experiments/ndtictactoe`)

- Environment supports variable board size (`N x N`)
- Agents:
  - DQN with replay memory + target network
  - REINFORCE with optional critic/baseline behavior
- Supports:
  - Training
  - AI vs human
  - AI vs random policy
  - Hyperparameter sweep workflows (via `ttt_main.cpp`)

## Data and artifacts

This repository may contain large generated artifacts (especially RL model checkpoints):

- Most disk usage is under `experiments/ndtictactoe/models-and-data`
- Recommender also includes large CSV datasets and many JSON outputs

The root `.gitignore` ignores common build products, `*.model` checkpoints, and the **`experiments/ndtictactoe/models-and-data/`** tree so new artifacts are not added by mistake.

**If those paths were committed before `.gitignore`:** Git will keep tracking them until you remove them from the index (files stay on disk):

```bash
bash scripts/git-untrack-artifacts.sh
git status   # review
git commit -m "chore: stop tracking experiment artifacts"
```

That does **not** shrink old history; use `git filter-repo` (or similar) only if you need a smaller clone and are willing to rewrite public history.

Prefer writing new checkpoints under a local **`artifacts/`** directory (tracked only via `.gitkeep`) and running experiments from the directory where relative data paths resolve.

## Known limitations

- Some planned features in comments/proposal are partial or not productionized.
- `lib/math/token.hpp` lexer/parser scaffolding is incomplete.
- Several scripts/flows assume local environment details (Python version, headers, OS commands).
- Automated tests cover selected `lib` and DQN behaviors (`ctest` / `ctorch_tests`); coverage is not exhaustive.
- Documentation quality is uneven across source files.

## Contributing

- Configure and build from the repository root with CMake, then run **`ctest --test-dir build --output-on-failure`** before opening a pull request.
- Prefer small, focused changes; match existing style and include tests when you add or fix behavior.

## Kernel SVM and Random Fourier notes

- **Kernel SVM (RBF):** `kernelsvm` uses `exp(-gamma * d^2)` where `d` is the Euclidean distance between row vectors.
- **Random Fourier SVM:** Each row of `W` is drawn from a Gaussian with standard deviation `sqrt(2 * gamma)`, and each phase in `b` is uniform on `[0, 2π)`. That ties `gamma` to the usual RBF bandwidth in a Random Fourier-style approximation; results are educational and may differ from another library’s exact defaults.

## Future work

Examples aligned with the original proposal: richer automatic differentiation, additional models (e.g. trees, gradient boosting), better serialization and experiment configs, and broader test coverage for RL and recommender paths.

## Project proposal

The original project proposal is in:

- [`proposal/Project Proposal.md`](proposal/Project%20Proposal.md)
- [`proposal/Project Proposal.pdf`](proposal/Project%20Proposal.pdf)
