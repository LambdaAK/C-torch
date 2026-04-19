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

For notebook usage, set `CTORCH_LIB_PATH` before importing `ctorch` if the shared library is outside the package directory.
