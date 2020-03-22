from abc import abstractmethod, ABC
from collections import defaultdict
from typing import Tuple, Optional

import numpy as np


class Computation(ABC):
    def __init__(self,
                 shape: Optional[Tuple[int, ...]] = None,
                 requires_grad: bool = False):
        self.gradient = None
        self.shape = shape
        self.requires_grad = requires_grad
        self.references = defaultdict(lambda: 0)
        self.ctx = None

    def new_context(self):
        self.ctx = self.references.copy()

    def accumulate(self, reference: 'Computation', gradient: np.array):
        if self.ctx[reference] <= 0:
            raise ValueError('Invalid node for gradient propagation')
        else:
            self.ctx[reference] -= 1
        if self.shape is None:
            self.shape = gradient.shape
            self.gradient = gradient
        else:
            assert gradient.shape == self.gradient.shape
            self.gradient += gradient

    def register(self, reference: 'Computation'):
        # Add a permanent computational node (independent of context)
        self.references[reference] += 1

    def unbroadcast(self,
                    matrix: np.ndarray,
                    shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        shape = shape if shape else self.shape
        while matrix.ndim < len(shape):
            matrix = np.expand_dims(matrix,0)

        for ((n,i),j) in zip(enumerate(matrix.shape), shape):
            if i == j:
                continue
            elif i == 1:
                # j!=1 -> Expand matrix
                new_shape = list(matrix.shape)
                new_shape[n] = j
                matrix = np.broadcast_to(matrix, new_shape)
            elif j == 1:
                # i!=1 -> Sum and reshape matrix (since np.sum removes 1 dimension)
                new_shape = list(matrix.shape)
                new_shape[n] = 1
                matrix = matrix.sum(n).reshape(new_shape)
            else:
                raise ValueError(f'Cannot broadcast shape {matrix.shape} to {shape}')
        return matrix

    @abstractmethod
    def forward(self):
        pass