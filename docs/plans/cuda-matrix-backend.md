# Plan: optional CUDA backend for `Matrix`

**Goal:** the core `Matrix` ops (matmul, add/sub, scalar mul, elementwise product,
transpose, relu/sigmoid/tanh and their derivatives) run on an NVIDIA GPU when one is
available, with no change to any caller. Every model, the NN stack and the Python
bindings speed up automatically because they all go through `Matrix`.

**Time box:** ~20 minutes. Out of scope: a PyTorch-style `.to("cuda")` device tensor,
GPU-side memory between ops, multi-GPU, cuDNN, and autograd changes.

## Key constraint

This machine is an Apple M3 Max: there is no NVIDIA GPU and no `nvcc`, and CUDA does
not run on macOS. So:

- CUDA is **off by default** (`CTORCH_ENABLE_CUDA=OFF`). The default Mac build must stay
  byte-for-byte unaffected.
- The CUDA path can only be compiled and tested on a Linux/Windows machine with an
  NVIDIA GPU (a cloud GPU box, Colab, a lab machine).
- Locally we can verify only that the default build and tests still pass.

## Design (smallest thing that works)

The data stays in `std::vector<double>` on the host. Each GPU-backed op does:
copy in → kernel → copy out. That's simple, has no new ownership or lifetime rules, and
makes parity tests trivial. The copies are the main cost, so ops route to the GPU
only above a size threshold. Matmul (O(n³) compute over O(n²) data) is the op that
actually benefits.

```
Matrix::operator*(Matrix)  ──► if cuda::should_use(work) ──► cuda::matmul(...)
                            └─► else existing parallel_for CPU path (unchanged)
```

- **Precision:** keep `double` to match the CPU results exactly. Consumer GPUs run
  FP64 slowly. Matmul via cuBLAS `cublasDgemm` is still much faster than the CPU
  for large n. Switching to float is a later decision.
- **Runtime switch:** `ctorch::cuda::set_enabled(bool)` and `is_available()`, plus the
  env var `CTORCH_DEVICE=cpu` to force the CPU path. Default: use the GPU if the build
  has CUDA and a device exists.

## Files

| File | Change |
|---|---|
| `lib/math/cuda_backend.hpp` | **New.** API: `is_available()`, `set_enabled()`, `should_use(size_t work)`, plus one function per op taking raw `const double*` buffers and dims. |
| `lib/math/cuda_backend.cu` | **New.** Built only when CUDA is on. Matmul via `cublasDgemm`, with row-major handled by swapping operands (compute `Bᵀ·Aᵀ` in column-major). One small grid-stride kernel template covers all the elementwise/unary ops. Transpose is a simple kernel. Errors go through a `CUDA_CHECK` macro that throws `std::runtime_error`. |
| `lib/math/cuda_backend_stub.cpp` | **New.** Built when CUDA is off. `is_available()` returns false, so `should_use()` is always false. The op functions are never called, so they just throw. |
| `lib/math/matrix.cpp` | Add a 3-line early-return dispatch at the top of `transpose`, `+`, `-`, `*(double)`, `*(Matrix)`, `elm_wise_product`, `relu`, `sigmoid`, `tanh` and their `_deriv` versions. The shape checks stay before the dispatch. |
| `CMakeLists.txt` | `option(CTORCH_ENABLE_CUDA ... OFF)`. If ON: `enable_language(CUDA)`, `find_package(CUDAToolkit REQUIRED)`, add the `.cu` to `ctorch`, link `CUDA::cudart CUDA::cublas`, define `CTORCH_HAS_CUDA`, set `CUDA_ARCHITECTURES native`. If OFF: add the stub. |
| `tests/test_cuda_matrix.cpp` | **New.** Parity tests: random matrices (including odd sizes like 1×n, 37×53, and 512×512), GPU result vs. CPU result within 1e-9 for each op. `GTEST_SKIP()` when `!is_available()`. |
| `Makefile`, `README.md` | `make build-cuda` target and a short "CUDA (optional)" section. |
| `python/ctorch/__init__.py` | *(Optional, ~2 min.)* Expose `cuda_is_available()` / `set_cuda_enabled()` through 2 new C API functions. |

## Steps (~20 min)

1. **Backend header + stub, wired into CMake (CUDA off).** Build and run `make test` on
   the Mac. Everything should pass unchanged. *(4 min)*
2. **Dispatch hooks in `matrix.cpp`.** With the stub these are no-ops. Rebuild and
   retest on the Mac. *(4 min)*
3. **Write `cuda_backend.cu`**: cuBLAS matmul, one elementwise kernel template
   (add/sub/scale/mul/relu/sigmoid/tanh and derivs), and a transpose kernel. *(6 min)*
4. **Parity test file.** On the Mac it compiles and skips. *(3 min)*
5. **Makefile/README** (plus the optional Python toggle). *(3 min)*
6. **Verify on a GPU machine** (outside the 20 min):
   `cmake -B build -DCTORCH_ENABLE_CUDA=ON && cmake --build build && ctest --test-dir build`.
   Then time a 1024×1024 matmul with the CPU vs. the GPU to tune the threshold.

## Done means

- The default Mac build and all existing tests pass, with no behavior change.
- With `-DCTORCH_ENABLE_CUDA=ON` on an NVIDIA box, the parity tests pass and large
  matmuls are visibly faster.
- No caller code changes anywhere (models, NN, bindings).

## Natural follow-ups (not in this plan)

- A device-resident tensor (`Matrix::to(Device::CUDA)`) so NN forward/backward stays on
  the GPU instead of copying every op. That's the real PyTorch-style step.
- FP32 option; fusing bias-add + activation in `LinearLayer`.
- A **Metal (MPS) backend** instead, which would actually run on this Mac.
