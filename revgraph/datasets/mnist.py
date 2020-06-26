import gzip
import os
from typing import Dict, Union, Tuple

import numpy as np


# Source: http://yann.lecun.com/exdb/mnist/


def load_data(as_dict: bool = False,
              one_hot: bool = True,
              normalize: bool = True):
    base_path = os.path.dirname(os.path.abspath(__file__))
    x_train_path = os.path.join(base_path, 'data/mnist/train-images-idx3-ubyte.gz')
    y_train_path = os.path.join(base_path, 'data/mnist/train-labels-idx1-ubyte.gz')
    x_test_path = os.path.join(base_path, 'data/mnist/t10k-images-idx3-ubyte.gz')
    y_test_path = os.path.join(base_path, 'data/mnist/t10k-labels-idx1-ubyte.gz')
    x_train = read_image(x_train_path, 60000)
    y_train = read_label(y_train_path, 60000)
    x_test = read_image(x_test_path, 10000)
    y_test = read_label(y_test_path, 10000)

    assert x_train.shape == (60000, 784)
    assert y_train.shape == (60000,)
    assert x_test.shape == (10000, 784)
    assert y_test.shape == (10000,)

    if normalize:
        x_train /= 255.0
        x_test /= 255.0

    if one_hot:
        n_categories = 10
        y_train = one_hot_encode(y_train, n_categories)
        y_test = one_hot_encode(y_test, n_categories)

    if as_dict:
        return {
            'train': {'x': x_train, 'y': y_train},
            'test': {'x': x_test, 'y': y_test}
        }
    return (x_train, y_train), (x_test, y_test)


def read_image(path: str, n_records: int = 60000) -> np.ndarray:
    with gzip.open(path, 'r') as fs:
        w = h = 28
        fs.read(16)  # Ignore preamble
        buffer = fs.read(w * h * n_records)
        return (np.frombuffer(buffer, dtype=np.uint8)
                  .astype(np.float64)
                  .reshape(n_records, w * h))


def read_label(path: str, n_records: int = 60000) -> np.ndarray:
    with gzip.open(path, 'r') as fs:
        fs.read(8)  # Ignore preamble
        return (np.array([np.frombuffer(fs.read(1), dtype=np.uint8)
                            .astype(np.float64)
                          for _ in range(n_records)])
                  .reshape(n_records))


def one_hot_encode(y: np.ndarray, n_categories: int) -> np.ndarray:
    n, = y.shape
    new_y = np.zeros((n, n_categories))
    for i in range(n):
        new_y[i][y[i].astype(np.int64)] = 1.0
    return new_y
