# C-torch

A from-scratch C++ machine learning project.  
The repository contains:

- A lightweight math layer (`Matrix`, symbolic AST expressions, numerical optimization helpers)
- ML algorithms implemented directly in C++
- End-to-end experiments for classification, recommendation, and reinforcement learning

This codebase is primarily experiment-driven and educational/research oriented.

## Table of contents

1. [Repository layout](#repository-layout)
2. [Implemented components](#implemented-components)
3. [Requirements](#requirements)
4. [Build and run](#build-and-run)
5. [Experiments](#experiments)
6. [Data and artifacts](#data-and-artifacts)
7. [Known limitations](#known-limitations)
8. [Project proposal](#project-proposal)

## Repository layout

```text
.
├── Makefile            # convenience: make build / make test
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
  - Kernel SVM (`kernelsvm.*`)
  - Random Fourier SVM (`randomfouriersvm.*`)
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

### Make (per experiment)

Each experiment directory still has a `Makefile` for direct `g++` builds.

### 1) Classification (Iris)

Run from `experiments/classification` so `Iris.csv` resolves.

```bash
cd experiments/classification
make
./main
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

- Loads `Iris.csv`
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

## Project proposal

The original project proposal is in:

- [`proposal/Project Proposal.md`](proposal/Project%20Proposal.md)
- [`proposal/Project Proposal.pdf`](proposal/Project%20Proposal.pdf)
