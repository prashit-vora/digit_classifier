# MNIST Neural Network from Scratch

This project implements a basic neural network to classify handwritten digits (MNIST dataset) using only NumPy. No machine learning frameworks were used — just raw matrix math and gradient descent.

## Features

- Single hidden-layer neural network
- ReLU and Softmax activation functions
- Cross-entropy loss
- SGD training with mini-batch updates
- Achieves ~91% accuracy on MNIST test data (trained on 10,000 samples)

## Dataset
The model uses the [MNIST dataset in CSV format](https://github.com/phoebetronic/mnist/raw/main/mnist_train.csv.zip). Make sure to download and extract both `mnist_train.csv` and `mnist_test.csv`.

- `mnist_train.csv`
- `mnist_test.csv`

Place both files in the project root before training.

## How to Run

```bash
python3 digit_classifier.py
