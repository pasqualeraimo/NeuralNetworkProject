from typing import Any

import numpy as np
import tensorflow as tf
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit


def load_mnist(train_size: int=48000,
               val_size: int=12000,
               test_size: int=10000)-> tuple[
    tuple[Any, ndarray[Any, dtype[floating[_64Bit]]]], tuple[Any, ndarray[Any, dtype[floating[_64Bit]]]], tuple[
        Any, ndarray[Any, dtype[floating[_64Bit]]]]]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).T
    y_train = np.eye(10)[np.array(y_train)].T
    x_training_set = x_train[:, :train_size]  # Primi 10000 campioni per il training set
    x_validation_set = x_train[:, train_size:train_size + val_size]  # Successivi 10000 campioni per il validation set
    y_training_set = y_train[:, :train_size]
    y_validation_set = y_train[:, train_size:train_size + val_size]
    x_test_set = x_test.reshape(test_size, 784).T
    y_test_set = np.eye(10)[np.array(y_test)].T
    x_test_set = x_test_set[:, :test_size]
    y_test_set = y_test_set[:, :test_size]
    return (x_training_set, y_training_set), (x_validation_set, y_validation_set), (x_test_set, y_test_set)