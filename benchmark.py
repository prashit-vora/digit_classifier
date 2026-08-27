import argparse
import platform
import time
from pathlib import Path

import numpy as np

from digit_classifier import MLPClassifier, load_mnist_csv


CONFIGURATIONS = [
    ("one layer", [64]),
    ("two layers", [128, 64]),
    ("wider two layers", [256, 128]),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark several NumPy MLP sizes.")
    parser.add_argument("--train-file", type=Path, default=Path("mnist_train.csv"))
    parser.add_argument("--test-file", type=Path, default=Path("mnist_test.csv"))
    parser.add_argument("--train-samples", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    x_train, y_train, x_test, y_test = load_mnist_csv(
        args.train_file, args.test_file, args.train_samples
    )

    print(f"Python {platform.python_version()}, NumPy {np.__version__}")
    print(
        f"{x_train.shape[0]:,} training samples, {args.epochs} epochs, "
        f"batch size {args.batch_size}, seed {args.seed}\n"
    )
    print("| Model | Hidden layers | Parameters | Test accuracy | Train time |")
    print("| --- | ---: | ---: | ---: | ---: |")

    for name, hidden_sizes in CONFIGURATIONS:
        model = MLPClassifier(
            input_size=x_train.shape[1],
            hidden_sizes=hidden_sizes,
            output_size=10,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )

        started = time.perf_counter()
        model.fit(
            x_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=False,
        )
        elapsed = time.perf_counter() - started
        _, accuracy = model.evaluate(x_test, y_test)

        layers = " x ".join(map(str, hidden_sizes))
        print(
            f"| {name} | {layers} | {model.parameter_count:,} | "
            f"{accuracy:.2%} | {elapsed:.2f}s |"
        )


if __name__ == "__main__":
    main()
