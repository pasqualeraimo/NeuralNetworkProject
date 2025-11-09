# NeuralNetworkProject

## Overview
This repository contains a from-scratch implementation of a fully connected neural network designed as a university project to classify handwritten digits from the [MNIST](https://yann.lecun.com/exdb/mnist/) dataset. The codebase avoids high-level deep-learning frameworks for the core learning loop, focusing instead on understanding every computational step: weight initialization, forward propagation, backpropagation, optimization, and early-stopping strategies.

The project provides:
- A modular neural network class with configurable topology and activation functions.
- Multiple loss definitions, including cross-entropy tailored for softmax outputs.
- Support for both Stochastic Gradient Descent (SGD) and the resilient backpropagation (RProp) optimization algorithm.
- Built-in early-stopping heuristics based on generalization loss and progress quotient.
- Experiment scripts that reproduce the evaluation pipeline and export plots/statistics.

## Repository structure
| Path | Description |
| ---- | ----------- |
| `activation_functions.py` | Element-wise activation functions (sigmoid, ReLU, tanh, identity) with derivative support. |
| `error_functions.py` | Loss functions (sum of squares, cross-entropy, cross-entropy with softmax) and their derivatives. |
| `mnist_loader.py` | TensorFlow-based helper that downloads and preprocesses MNIST into train/validation/test splits with one-hot labels. |
| `neural_network.py` | Core implementation of the dense neural network, including forward/backward passes, SGD/RProp updates, early stopping, and accuracy evaluation. |
| `network_types.py` | Type aliases for activation and error function signatures used throughout the codebase. |
| `test.py` | Experiment driver that orchestrates training runs across hyperparameters, records metrics, and generates plots under the `results/` directory. |
