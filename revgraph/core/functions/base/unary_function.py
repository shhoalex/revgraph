from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from revgraph.core.base.computation import Computation
from revgraph.core.base.function import Function


class UnaryFunction(Function, ABC):
    def __init__(self,
                 a: Computation,
                 shape: Tuple[int, ...] = None):
        super(UnaryFunction, self).__init__(args=[a], shape=shape)
        self.a = self.args[0]
        self.a_res = None

    def forward(self) -> np.ndarray:
        self.a_res = self.a.forward()
        return self.apply(a=self.a_res)

    def backward(self) -> None:
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