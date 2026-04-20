# Demos

Each subfolder here is a standalone demonstration program built against `ctorch`.

Current demos:

- `kernel/` - Kernel SVM on randomly generated concentric circles
  - `main.cpp` - native C++ demo
  - `main.py` - Python bindings demo
- `linear_regression/` - linear regression on a noisy line
  - `main.cpp` - native C++ demo
  - `main.py` - Python bindings demo
- `logistic_regression/` - logistic regression on two Gaussian blobs
  - `main.cpp` - native C++ demo
  - `main.py` - Python bindings demo

Build the demo from the repository root with:

```bash
make kernel-demo
make linear-regression-demo
make logistic-regression-demo
```

Or directly with CMake:

```bash
cmake --build build --target kernel_circle_demo --parallel
cmake --build build --target linear_regression_demo --parallel
cmake --build build --target logistic_regression_demo --parallel
```

After building, run the executable at `build/demos/kernel/kernel_circle_demo`.
The linear regression executable is at `build/demos/linear_regression/linear_regression_demo`.
The logistic regression executable is at `build/demos/logistic_regression/logistic_regression_demo`.

Run the Python version with:

```bash
python3 demos/kernel/main.py
python3 demos/linear_regression/main.py
python3 demos/logistic_regression/main.py
```

If the shared library has not been built yet, run `make py-bindings` first.

Each Python demo saves a PNG into `artifacts/plots/`:

- `kernel_svm_boundary.png`
- `linear_regression_fit.png`
- `logistic_regression_boundary.png`
