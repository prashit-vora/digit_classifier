import argparse
import time
from pathlib import Path

import numpy as np


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    return x > 0


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy(probabilities, labels):
    selected = probabilities[np.arange(labels.size), labels]
    return float(-np.mean(np.log(np.clip(selected, 1e-12, 1.0))))


def load_mnist_csv(train_file, test_file, train_samples=None):
    train = np.loadtxt(train_file, delimiter=",", dtype=np.uint8)
    test = np.loadtxt(test_file, delimiter=",", dtype=np.uint8)

    x_train = train[:, 1:].astype(np.float32) / 255.0
    y_train = train[:, 0].astype(np.int64)
    x_test = test[:, 1:].astype(np.float32) / 255.0
    y_test = test[:, 0].astype(np.int64)

    if train_samples is not None:
        x_train = x_train[:train_samples]
        y_train = y_train[:train_samples]

    return x_train, y_train, x_test, y_test


class MLPClassifier:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.1, seed=0):
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        sizes = [input_size, *hidden_sizes, output_size]
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            scale = np.sqrt(2.0 / fan_in)
            weights = self.rng.normal(0.0, scale, (fan_in, fan_out)).astype(np.float32)
            self.weights.append(weights)
            self.biases.append(np.zeros((1, fan_out), dtype=np.float32))

    @property
    def parameter_count(self):
        return sum(weight.size + bias.size for weight, bias in zip(self.weights, self.biases))

    def forward(self, x):
        activations = [x]
        pre_activations = []

        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = activations[-1] @ weight + bias
            pre_activations.append(z)
            activations.append(relu(z))

        logits = activations[-1] @ self.weights[-1] + self.biases[-1]
        probabilities = softmax(logits)
        return probabilities, activations, pre_activations

    def train_batch(self, x, labels):
        probabilities, activations, pre_activations = self.forward(x)
        delta = probabilities.copy()
        delta[np.arange(labels.size), labels] -= 1
        delta /= labels.size

        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.biases)

        for layer in range(len(self.weights) - 1, -1, -1):
            weight_gradients[layer] = activations[layer].T @ delta
            bias_gradients[layer] = np.sum(delta, axis=0, keepdims=True)
            if layer > 0:
                delta = (delta @ self.weights[layer].T) * relu_derivative(
                    pre_activations[layer - 1]
                )

        for layer in range(len(self.weights)):
            self.weights[layer] -= self.learning_rate * weight_gradients[layer]
            self.biases[layer] -= self.learning_rate * bias_gradients[layer]

        return cross_entropy(probabilities, labels)

    def predict_proba(self, x):
        probabilities, _, _ = self.forward(x)
        return probabilities

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    def evaluate(self, x, labels):
        probabilities = self.predict_proba(x)
        predictions = np.argmax(probabilities, axis=1)
        return cross_entropy(probabilities, labels), float(np.mean(predictions == labels))

    def fit(self, x, labels, epochs=5, batch_size=64, verbose=True):
        history = []
        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(labels.size)
            started = time.perf_counter()

            for start in range(0, labels.size, batch_size):
                indices = order[start : start + batch_size]
                self.train_batch(x[indices], labels[indices])

            loss, accuracy = self.evaluate(x, labels)
            duration = time.perf_counter() - started
            history.append({"loss": loss, "accuracy": accuracy, "seconds": duration})

            if verbose:
                print(
                    f"epoch {epoch:02d}/{epochs}: "
                    f"loss={loss:.4f} accuracy={accuracy:.2%} time={duration:.2f}s"
                )

        return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train a NumPy MLP on MNIST CSV files.")
    parser.add_argument("--train-file", type=Path, default=Path("mnist_train.csv"))
    parser.add_argument("--test-file", type=Path, default=Path("mnist_test.csv"))
    parser.add_argument("--train-samples", type=int, default=10_000)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 64])
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

    model = MLPClassifier(
        input_size=x_train.shape[1],
        hidden_sizes=args.hidden_sizes,
        output_size=10,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    architecture = " -> ".join(map(str, [x_train.shape[1], *args.hidden_sizes, 10]))
    print(f"training samples: {x_train.shape[0]:,}")
    print(f"architecture: {architecture}")
    print(f"parameters: {model.parameter_count:,}")
    started = time.perf_counter()
    model.fit(x_train, y_train, args.epochs, args.batch_size)
    elapsed = time.perf_counter() - started

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"test loss: {test_loss:.4f}")
    print(f"test accuracy: {test_accuracy:.2%}")
    print(f"training time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
