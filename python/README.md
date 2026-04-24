# ctorch Python Package

`ctorch` exposes typed Python bindings over the C-torch C++ implementation via `ctypes`.

## Install (editable)

From the repository root:

```bash
cmake -B build -DCTORCH_BUILD_TESTS=OFF -DCTORCH_BUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ctorch_c --parallel
CTORCH_LIB_PATH="$PWD/build/libctorch_c.dylib" python3 -m pip install -e .
```

On Linux, use `libctorch_c.so` instead of `libctorch_c.dylib`.

## Build a wheel

```bash
make py-wheel
```

This target:

1. Builds `ctorch_c` via CMake
2. Copies the shared library into `python/ctorch/`
3. Builds a wheel into `dist/`

## Typing support

This package ships a `py.typed` marker (PEP 561). Type checkers can consume inline annotations directly:

```python
from ctorch import Matrix, Sequential

x: Matrix = Matrix([[1.0], [2.0]])
model: Sequential = Sequential().add_linear(2, 1)
```

## Train/infer workflow

Use the typed workflow example:

```bash
python3 examples/python/train_infer_sequential.py train --model-path artifacts/py_models/line.model
python3 examples/python/train_infer_sequential.py infer --model-path artifacts/py_models/line.model --x 1.75
```

For a small nonlinear neural-network demo:

```bash
make py FILE=examples/python/train_nn_regression.py
python3 examples/python/train_nn_regression.py infer --model-path artifacts/py_models/nonlinear_regression.model --x0 0.25 --x1 -0.75
```

For a simple XOR-style distributed classification demo:

```bash
make py FILE=examples/python/train_xor_distributed.py
make py FILE=examples/python/train_xor_distributed.py ARGS="--distributed --rank 0 --world-size 2"
make py FILE=examples/python/train_xor_distributed.py ARGS="--distributed --rank 1 --world-size 2"
```

Distributed mode:

```bash
make py FILE=examples/python/train_nn_regression.py ARGS="--distributed --rank 0 --world-size 2"
make py FILE=examples/python/train_nn_regression.py ARGS="--distributed --rank 1 --world-size 2"
```

Add `--checkpoint-prefix artifacts/py_models/nonlinear_regression` to save distributed checkpoints, and `--resume-checkpoint` to reload them.
Start rank `0` first so the TCP master is listening before the other rank connects.

Classical models also expose clean distributed classmethods:

```python
from ctorch import LogisticRegression, OptimType, TcpProcessGroup

group = TcpProcessGroup("127.0.0.1", 29500, rank=0, world_size=2)
model = LogisticRegression.train_distributed(
    x_train,
    y_train,
    optim_type=OptimType.GD,
    learning_rate=0.05,
    max_iter=1000,
    group=group,
)
```

`LinearRegression`, `SVM`, `KernelSVM`, and `RandomFourierSVM` expose the same pattern with their own constructor arguments.

For notebook usage, set `CTORCH_LIB_PATH` before importing `ctorch` if the shared library is outside the package directory.
