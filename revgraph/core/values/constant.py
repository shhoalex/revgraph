from typing import Union

import numpy as np

from revgraph.core.base.value import Value


class Constant(Value):
    def __init__(self,
                 data: Union[np.ndarray, list]):
        self.assigned = False
        super(Constant, self).__init__(data=data,
                                       requires_grad=False)

    def accumulate(self, reference, _):
        raise ValueError('Constant cannot accumulate gradient')

    def register(self, _):
        raise ValueError('Constant cannot register parent node')

    def context_completed(self):
        return True

    @Value.data.setter
    def data(self, data):
        if not self.assigned:
            self.assigned = True
            super(Constant, self.__class__).data.fset(self, data)
        else:
            raise ValueError('Constant cannot be mutated')
