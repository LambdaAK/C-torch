"""
Linear SVM classification example with more complex data, higher dimensions, and feature augmentation.

From the repo root, after building the ctypes library:

    make py FILE=svm_example.py

or:

    PYTHONPATH="$PWD/python" python3 svm_example.py
"""

import random
from ctorch import DataAugmentationType, SVM


def generate_data(n_per_class=15, seed=123):
    """
    Generates a 2D dataset, two classes, with overlap and shift (not linearly separable).
    Also generates a simple outlier and a cluster offset for +1 class.
    """
    random.seed(seed)
    x_train, y_train = [], []

    # Class -1: cluster near (0, 0)
    for _ in range(n_per_class):
        x = [random.gauss(0.0, 0.4), random.gauss(0.0, 0.4)]
        x_train.append(x)
        y_train.append(-1.0)
    # Class +1: cluster near (1, 1) plus a few dispersed samples
    for _ in range(n_per_class):
        if _ < n_per_class - 2:
            x = [random.gauss(1.1, 0.5), random.gauss(1.1, 0.5)]
        else:
            # Add a difficult point or "outlier"
            x = [random.uniform(-0.7, +1.7), random.uniform(-0.7, +1.7)]
        x_train.append(x)
        y_train.append(1.0)
    return x_train, [y_train]

def main() -> None:
    # Generate a more challenging dataset
    x_train, y_train = generate_data()

    # Add a third feature using a nonlinear transform (to demo effect of augmentation)
    x_train_aug = [row + [row[0] ** 2 + row[1] ** 2] for row in x_train]

    # Try both no augmentation and polynomial (quadratic)
    for aug_type in [DataAugmentationType.NO_OP, DataAugmentationType.POLY_2]:
        print(f"Training SVM with augmentation type: {aug_type.name}")

        model = SVM(
            x_train_aug if aug_type == DataAugmentationType.NO_OP else x_train,
            y_train,
            learning_rate=0.05,
            max_iter=600,
            c_value=2.0,
            augmentation=aug_type,
        )

        # Test data: sample inside each cluster and ambiguous region
        x_test = [
            [0.1, 0.0, 0.1**2 + 0.0**2],
            [1.2, 1.3, 1.2**2 + 1.3**2],
            [0.65, 0.4, 0.65**2 + 0.4**2],      # In between
            [-0.5, -0.5, (-0.5)**2 + (-0.5)**2], # Outlier
            [1.5, 1.0, 1.5**2 + 1.0**2]
        ]
        y_test = [[-1.0, 1.0, 1.0, -1.0, 1.0]]

        preds = [model.predict([row]) for row in x_test]
        print(" Test samples:")
        for xi, yi, pi in zip(x_test, y_test[0], preds):
            print(f"  x={xi[:2]}   label={int(yi):+2d}   predicted={int(pi):+2d}")
        acc = model.score(x_test, y_test)
        print(f" Accuracy (fraction correct): {acc:.3f}\n")

if __name__ == "__main__":
    main()
