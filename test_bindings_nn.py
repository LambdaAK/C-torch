"""Neural-network binding smoke test: fit y = 3x + 1 with a tiny net, print random checks.

Run:
  PYTHONPATH="$PWD/python" python3 test_bindings_nn.py
"""

from __future__ import annotations

import random

from ctorch import Matrix, NNOptimType, NNOptimizer, Sequential


def true_y(x: float) -> float:
    return 15.4 * x + 1.2


def test_nn_fits_line() -> None:
    random.seed(42)

    # Training pairs (x, y) with x as a (1, 1) column — layout expected by the C++ linear layer.
    samples: list[tuple[list[list[float]], float]] = []
    for _ in range(50):
        x = random.uniform(-2.0, 3.0)
        samples.append(([[x]], true_y(x)))

    net = Sequential()
    net.add_linear(1, 1)  # y_hat = w * x + b (enough for a line)
    opt = NNOptimizer(net, NNOptimType.SGD, learning_rate=0.15, batch_size=len(samples))

    for _ in range(1000):
        opt.zero_grad()
        for xv, y in samples:
            y_hat = net.forward(Matrix(xv)).get(0, 0)
            # MSE 0.5*(y_hat - y)^2  ->  dL/dy_hat = y_hat - y
            net.backward(Matrix([[y_hat - y]]))
        opt.step()

    print("random x  |  expected (3x+1)  |  network output")
    print("-" * 52)
    for _ in range(6):
        x = random.uniform(-2.0, 3.0)
        got = net.forward(Matrix([[x]])).get(0, 0)
        exp = true_y(x)
        print(f"  {x:+.4f}  |  {exp:+.6f}        |  {got:+.6f}")
        assert abs(got - exp) < 0.05, f"too far from line at x={x}: got {got}, expected {exp}"


if __name__ == "__main__":
    test_nn_fits_line()
    print("ok")
