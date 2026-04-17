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
├── README.md
├── proposal/
│   ├── Project Proposal.md
│   └── Project Proposal.pdf
└── src/
    ├── math/           # matrix ops, AST, differentiator, optimizers, augmentation
    ├── ml/             # ML models and utilities
    └── experiments/
        ├── classification/
        ├── recommender/
        └── ndtictactoe/
```

## Implemented components

### `src/math`

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

### `src/ml`

- Supervised:
  - Perceptron (`perceptron.*`)
  - Linear SVM (`svm.*`)
  - Kernel SVM (`kernelsvm.*`)
  - Random Fourier SVM (`randomfouriersvm.*`)
  - Logistic Regression (`logisticregression.*`)
  - Linear Regression (`linearregression.*`)
  - KNN (`knn.*`)
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

- C++17+ compiler (`g++` used by provided Makefiles)
- `make`

### Recommender experiment extras

- `libcurl`
- Python 3.11 development headers/libs
- NumPy include path available to Python
- `matplotlibcpp.h` is vendored, but `nlohmann/json` single-header include path is expected by the Makefile/code (`json/single_include/...`)

### Notes

- The project is Linux/macOS oriented (system calls such as `open`/`xdg-open` are used in places).
- Paths are mostly relative to each experiment directory.

## Build and run

Each experiment has its own Makefile.

### 1) Classification (Iris)

```bash
cd src/experiments/classification
make
./main
```

### 2) Recommender (KMeans + PCA + MAB/UCB)

```bash
cd src/experiments/recommender
make
./main
```

### 3) N-dimensional Tic-Tac-Toe RL (DQN/REINFORCE)

```bash
cd src/experiments/ndtictactoe
make
./tictactoe
```

There is also an alternate RL driver:

```bash
cd src/experiments/ndtictactoe
g++ -std=c++20 -O3 ttt_main.cpp tictactoe.cpp replaymemory.cpp dqn.cpp sample.cpp ../../math/matrix.cpp ../../ml/nn.cpp -o ttt_main
./ttt_main
```

## Experiments

### Classification (`src/experiments/classification`)

- Loads `Iris.csv`
- Performs train/test split and feature normalization
- Runs:
  - Linear SVM
  - Perceptron
  - KNN
  - Linear regression scoring variant
  - Feedforward NN classifier

### Recommender (`src/experiments/recommender`)

- Loads `processed_data.csv`
- Uses selected audio features to:
  - Optionally reduce dimension with PCA
  - Cluster tracks with KMeans
  - Treat clusters as arms for MAB/UCB
- Simulated reward is based on cosine similarity to a user preference vector
- Writes experiment metrics to JSON result files
- Includes optional Spotify track/image lookup utilities (requires auth token handling in code)

### N-D Tic-Tac-Toe RL (`src/experiments/ndtictactoe`)

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

This repository currently contains large generated artifacts (especially RL model checkpoints):

- Most disk usage is under `src/experiments/ndtictactoe/models-and-data`
- Recommender also includes large CSV datasets and many JSON outputs

If you want a lightweight dev version, consider pruning `.model` files and old experiment outputs before cloning/sharing.

## Known limitations

- Some planned features in comments/proposal are partial or not productionized.
- `src/math/token.hpp` lexer/parser scaffolding is incomplete.
- Several scripts/flows assume local environment details (Python version, headers, OS commands).
- Minimal automated test coverage is included in-repo.
- Documentation quality is uneven across source files.

## Project proposal

The original project proposal is in:

- [`proposal/Project Proposal.md`](proposal/Project%20Proposal.md)
- [`proposal/Project Proposal.pdf`](proposal/Project%20Proposal.pdf)
