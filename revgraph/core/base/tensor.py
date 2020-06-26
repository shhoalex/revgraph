from abc import abstractmethod, ABC
from collections import defaultdict
from typing import Tuple, Optional

import numpy as np


from .tensor_magic import TensorMagic


class Tensor(ABC, TensorMagic):
    """
    The most basic unit of computation in this library.
    """
    def __init__(self,
                 shape: Optional[Tuple[int, ...]] = None,
                 requires_grad: bool = False):
        """
        The shape can be None and requires_grad specifies whether this tensor
        requires gradient from parent nodes.
        """
        self.gradient = None
        self.shape = shape
        self.requires_grad = requires_grad
        self.references = defaultdict(lambda: 0)
        self.ctx = None
        self.ctx_counter = None

    def context_completed(self):
        """
        Check whether the tensor has received all gradients from all its parent
        nodes within a "gradient context".
        """
        return self.ctx_counter == 0

    def new_context(self):
        """
        Create a new "gradient context".
        """
        self.ctx = self.references.copy()
        self.ctx_counter = sum(self.ctx.values())

    def accumulate(self, reference: 'Tensor', gradient: np.array):
        """
        Accumulate gradient FROM the reference to itself.
        """
        if self.ctx[reference] <= 0:
            raise ValueError('Invalid node for gradient propagation')
        else:
            self.ctx[reference] -= 1
            self.ctx_counter -= 1
        gradient = np.array(gradient)
        if self.shape is None:
            self.shape = gradient.shape
            self.gradient = gradient
        elif self.gradient is None:
            assert self.shape is not None
            self.gradient = self.unbroadcast(gradient)
        else:
            if gradient.shape == self.gradient.shape:
                self.gradient += gradient
            else:
                self.gradient += self.unbroadcast(gradient)

    def register(self, reference: 'Tensor'):
        """
        Add a permanent computational node (independent of context)
        """
        self.references[reference] += 1

    def unbroadcast(self,
                    matrix: np.ndarray,
                    shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """
        Try to match the matrix's shape with its own
        (same "unbroadcasting rule" as the one in autograd and numpy)
        Source: https://github.com/HIPS/autograd/blob/c6f630a5ec18bd30f1485bc0dbbccb8664c77510/autograd/numpy/numpy_vjps.py#L647
        """
        shape = shape if shape else self.shape
        if shape is ():
            return matrix.sum()
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
    def forward(self, *args, **kwargs):
        pass

    def clear_gradient(self):
        self.gradient = None

    def __repr__(self):
        return '<tensor object at {tid}>'.format(tid=hex(id(self)))

    def __str__(self):
        return 'tensor(addr={tid})'.format(tid=hex(id(self)))
