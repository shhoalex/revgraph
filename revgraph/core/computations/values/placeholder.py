from typing import Union, Tuple, Optional

import numpy as np

from revgraph.core.computations.base.value import Value


class Placeholder(Value):
    def __init__(self,
                 shape: Tuple[Optional[int], ...],
                 requires_grad: bool = False,
                 default: Optional[Union[np.ndarray, list]] = None,
                 name: Optional[str] = None):
        super(Placeholder, self).__init__(shape=shape,
                                          requires_grad=requires_grad)
        self.default = default
        self.name = name

    def valid_shape(self, data: np.ndarray):
        if data.ndim != len(self.shape):
            return False
        for i,j in zip(self.shape, data.shape):
            if i is not None and i != j:
                return False
        return True

    def feed(self,
             data: Optional[Union[np.ndarray, list]] = None) -> Value:
        if data is None and self.default is None:
            raise ValueError('Value not found')
        elif data is None:
            data = np.array(self.default)
        if not self.valid_shape(data):
            raise ValueError(f'Expected shape: {self.shape}, got {data.shape} instead')
        return Value(data=data, requires_grad=self.requires_grad)
