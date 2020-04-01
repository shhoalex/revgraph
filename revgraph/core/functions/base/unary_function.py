from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from revgraph.core.base.computation import Computation
from revgraph.core.base.function import Function


class UnaryFunction(Function, ABC):
    def __init__(self,
                 a: Union[Computation, list, int, float]):
        super(UnaryFunction, self).__init__(args=[a])
        self.a = self.args[0]
        self.a_res = None

    def forward(self, *args, **kwargs) -> np.ndarray:
        self.a_res = self.a.forward()
        self.data = self.apply(a=self.a_res)
        return self.data

    def backward(self, *args, **kwargs) -> None:
        if self.a.requires_grad:
            ga = self.gradient_wrt_a(gradient=self.gradient,
                                     a=self.a_res)
            self.a.accumulate(self, ga)

    @abstractmethod
    def apply(self, a: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_wrt_a(self,
                       gradient: np.ndarray,
                       a: np.ndarray) -> np.ndarray:
        pass
