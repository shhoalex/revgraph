from typing import Union, Tuple, Optional

import numpy as np

from revgraph.core.base.value import Value


class Placeholder(Value):
    """
    A placeholder represents a value in which the actual data won't be present
    until the actual execution.
    """
    def __init__(self,
                 name: str,
                 shape: Tuple[Optional[int], ...]):
        super(Placeholder, self).__init__(shape=shape)
        self.shape_constraint = shape
        self.name = name

    def valid_shape(self, data: np.ndarray):
        if data.ndim != len(self.shape):
            return False
        for i,j in zip(self.shape_constraint, data.shape):
            if i is not None and i != j:
                return False
        return True

    def clear_value(self):
        self.shape = self.shape_constraint
        self.data = None

    def feed(self,
             data: Union[np.ndarray, list, int, float]):
        data = np.array(data)
        if not self.valid_shape(data):
            raise ValueError(f'Expected shape: {self.shape}, got {data.shape} instead')
        super(Placeholder, self).__init__(data=data,
                                          shape=data.shape,
                                          requires_grad=False)
