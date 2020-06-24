from abc import ABC

import numpy as np

from revgraph.core.base.tensor import Tensor
from revgraph.core.base.value import Value
from .generic_function import GenericFunction


class NoGradFunction(GenericFunction, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.target_function = None
        for arg in args:
            if not isinstance(arg, Tensor):
                arg = Value(np.array(arg), requires_grad=False)
            self.args.append(arg)
        self.dependencies = {self}.union(*map(lambda n: n.dependencies, self.args))

    def backward(self, *args, **kwargs):
        pass

    def apply(self, *args, **kwargs):
        return self.target_function(*args, **kwargs)
