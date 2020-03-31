from abc import ABC, abstractmethod

import numpy as np

from revgraph.core.base.computation import Computation
from revgraph.core.base.value import Value
from .generic_function import GenericFunction


class NoGradFunction(GenericFunction, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        for arg in args:
            if not isinstance(arg, Computation):
                arg = Value(np.array(arg), requires_grad=False)
            self.args.append(arg)
        self.dependencies = set().union(*map(lambda n: n.dependencies, self.args))

    def backward(self, *args, **kwargs):
        pass

    def apply(self, *args, **kwargs):
        return self.__class__.call(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def call(*args, **kwargs):
        pass
