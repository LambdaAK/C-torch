# C-torch Bug Audit (2026-04-19)

## Scope
- Reviewed core math/ML libraries, RL experiment code, C API bindings, and representative experiment utilities.
- Ran existing tests, targeted Python repro scripts, and targeted C++ repro programs (including ASAN for memory safety cases).

## Exhaustive Bug List

| ID | Severity | Location | Bug | Evidence / Repro |
|---|---|---|---|---|
| B001 | Critical | `lib/math/matrix.cpp:12` | `Matrix(std::initializer_list<...>)` dereferences `init_list.begin()` without checking empty list. `Matrix({})` segfaults. | Confirmed: standalone repro exits `rc=139`. |
| B002 | High | `lib/math/dataaugmentor.cpp:85-113` | `random_fourier_features` accepts invalid `D` and `gamma` values. `D<0` overflows into huge allocation; `gamma<0` makes `sqrt(2*gamma)` invalid and outputs NaNs; `D==0` silently returns `n x 0`. | Confirmed: C++ repro shows `D-1 exception: vector`, `gamma=-1` yields `nan`. |
| B003 | High | `lib/ml/knn.cpp:11-18` | `KNN` constructor validates only `k>0`; it does not validate `xTr`/`yTr` sample-count compatibility or `yTr` row shape. Invalid model instances are accepted and fail later. | Confirmed via Python binding: model constructed with mismatched rows/labels, then `predict` fails with out-of-bounds. |
| B004 | Medium | `lib/ml/knn.cpp:83-95` | `KNN::predict` returns `-1` when training data is empty (`distances` empty). This is a silent invalid-class output instead of a validation error. | Confirmed via Python binding: empty train set predict returns `-1`. |
| B005 | High | `lib/ml/linearregression.cpp:129-143` | `LinearRegression::predict` loops over input columns only, with no shape check against model weights. If input has fewer features, prediction silently ignores learned weights. | Confirmed via Python binding: 2-feature model accepts 1-feature sample and returns numeric output without error. |
| B006 | High | `lib/ml/linearregression.cpp:145-175` | `LinearRegression::score` does not validate `xTe.numRows() == yTe.numCols()`. With mismatched counts, it can silently compute wrong accuracy (denominator from `yTe`, loop bound from `xTe`). | Confirmed via Python binding: mismatch returns numeric score instead of input-shape error. |
| B007 | Medium | `lib/ml/linearregression.cpp:65-84` | `LinearRegression` constructor lacks early shape validation; mismatches fail deep in loss construction (`Matrix::at`), producing unclear errors instead of explicit argument validation. | Confirmed via Python binding (`LinearRegression(...) failed: Matrix::at(): Index out of bounds`). |
| B008 | High | `lib/ml/pca.cpp:6-10` | PCA covariance scaling uses `1.0/(X.numRows()-1)` with no guard for `<2` samples. With one sample, covariance/projection become NaN. | Confirmed via Python binding: one-row input produces `[[nan], [nan]]` projection. |
| B009 | High | `lib/ml/kmeans.cpp:99-109` | Empty-cluster recovery is broken: random centroid is assigned to `this->centroids[cluster]`, then immediately overwritten by `centroids = std::move(new_centroids)` where `new_centroids[cluster]` stayed zero. | Static code proof. |
| B010 | High | `lib/ml/randomfouriersvm.cpp:13-27` | `RandomFourierSVM` constructor does not validate `D`/`gamma`; negative `D` leads allocation failure, invalid `gamma` yields invalid sampling stddev. | Confirmed via Python binding: `d_features=-1` fails with allocation exception (`vector`). |
| B011 | Medium | `lib/ml/randomfouriersvm.cpp:71-95` | `RandomFourierSVM::score` has no test-label shape checks before indexing `yTe.at(0,i)`. Invalid inputs fail late. | Static code proof. |
| B012 | Medium | `lib/bindings/ctorch_c_api.cpp:1202-1222` | `ctorch_random_fourier_svm_create` forwards `d_features`/`gamma` directly without front-door validation (unlike augmentation dispatch path). | Static code proof + runtime failure inherited from B010. |
| B013 | Critical | `lib/ml/mab.cpp:30-35` | `MAB::update` writes `counts[arm]` / `values[arm]` without bounds check. | Confirmed with ASAN: heap-buffer-overflow (abort `rc=134`). |
| B014 | Critical | `lib/ml/ucb.cpp:37-42` | `UCB::update` writes `counts[arm]` / `values[arm]` without bounds check. | Confirmed with ASAN: heap-buffer-overflow (abort `rc=134`). |
| B015 | High | `lib/ml/mab.cpp:6-27`, `lib/ml/ucb.cpp:10-35` | `n_arms <= 0` is accepted; selection logic then returns arm `0` even when no arms exist, yielding invalid action IDs and undefined policy behavior. | Confirmed via Python binding: `MAB(0,...)` and `UCB(0)` both construct and `select_arm()` returns `0`. |
| B016 | Critical | `experiments/ndtictactoe/replaymemory.cpp:6-10` | `ReplayMemory::add` pops from empty deque when `capacity==0` (`memory.pop_front()` on empty container). | Confirmed: standalone repro exits `rc=139`. |
| B017 | Critical | `experiments/ndtictactoe/reinforce.cpp:86-89` | `Reinforce::update_last_reward` assumes `episodes[batch_count]` non-empty and in-range; calling before a transition causes invalid `.back()` access. | Confirmed: standalone repro exits `rc=139`. |
| B018 | High | `experiments/ndtictactoe/dqn.cpp:166-168` | `steps % update_frequency` is executed without guarding `update_frequency > 0`; this is undefined behavior when zero is configured. | Static code proof (modulo-by-zero UB path). |
| B019 | High | `lib/ml/nn.cpp:344-396` | `Sequential::load` never checks `file.read()` success/failbit while reading weights/biases. Truncated binary files can load "successfully" and corrupt parameters. | Confirmed via Python binding: tail-truncated file loads without error. |
| B020 | Medium | `lib/ml/nn.cpp:82-99` | `LinearLayer` constructor does not validate positive dimensions. Negative dimensions become huge `size_t` allocations (`vector` exception), instead of explicit argument error. | Confirmed via standalone repro (`exception: vector`). |
| B021 | Medium | `experiments/ndtictactoe/tictactoe.hpp:85` | `TicTacToe::set_player_to_optimize_for(int)` is declared but not defined, causing linker failure when used. | Confirmed: standalone compile/link fails with undefined symbol. |
| B022 | Medium | `experiments/classification/main.cpp:105-127` | `get_random_batch` does not guard `batch_size <= total_samples`; indexing `indices[i]` can go out of bounds. | Static code proof. |

## Notes
- Existing `ctest` and Python binding tests pass, so these are largely uncovered edge-case and API-hardening defects.
- Several issues are undefined-behavior/memory-safety bugs and should be treated as priority fixes.
