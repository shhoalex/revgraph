from typing import Optional, Tuple, Union

import numpy as np

from .tensor import Tensor


class Value(Tensor):
    def __init__(self,
                 data: Optional[Union[np.ndarray, list]] = None,
                 shape: Optional[Tuple[Optional[int], ...]] = None,
                 requires_grad: bool = False):
        """
        A value represents tensor with the data property.
        """
        super(Value, self).__init__(shape, requires_grad)
        self.data = data
        self.dependencies = {self}

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        data = data if isinstance(data, np.ndarray) else np.array(data, dtype='float64')
        if self.shape is None:
            self.shape = data.shape
        else:
            for i,j in zip(self.shape, data.shape):
                if i is not None and i != j:
                    raise ValueError(f'Expected shape: {self.shape}, got {data.shape} instead')
        self._data = data

    def forward(self) -> 'np.ndarray':
        return self.data
