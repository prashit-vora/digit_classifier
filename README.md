# MNIST classifier in NumPy

A small multilayer perceptron for classifying MNIST digits. The forward pass,
backpropagation, mini-batch training, and cross-entropy loss are implemented
directly with NumPy.

The default model has two hidden layers with 128 and 64 units. PyTorch and
TensorFlow are not used.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download and extract the MNIST CSV files:

- [Training set](https://github.com/phoebetronic/mnist/raw/main/mnist_train.csv.zip)
- [Test set](https://github.com/phoebetronic/mnist/raw/main/mnist_test.csv.zip)

Place `mnist_train.csv` and `mnist_test.csv` in the repository root.

## Training

The default run uses 10,000 training examples for five epochs:

```bash
python3 digit_classifier.py
```

The model size and training settings can be changed from the command line:

```bash
python3 digit_classifier.py \
  --train-samples 60000 \
  --hidden-sizes 256 128 \
  --epochs 10 \
  --batch-size 64 \
  --seed 0
```

## Benchmarks

The benchmark compares three network sizes using the same data order and
initialization seed:

```bash
python3 benchmark.py
```

Results below were measured on an Intel Core i3-1115G4 using Python 3.12.3 and
NumPy 1.26.4. Each model used 10,000 training examples, five epochs, a batch
size of 64, and seed 0.

| Model | Hidden layers | Parameters | Test accuracy | Train time |
| --- | ---: | ---: | ---: | ---: |
| one layer | 64 | 50,890 | 92.05% | 7.85s |
| two layers | 128 x 64 | 109,386 | 92.01% | 14.87s |
| wider two layers | 256 x 128 | 235,146 | 91.03% | 30.39s |

Timing depends on the NumPy build and available CPU. Accuracy should be
reproducible with the same seed and dataset.

## Tests

```bash
python3 -m unittest discover -s tests
```
