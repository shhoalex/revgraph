from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from revgraph.core.base.computation import Computation
from revgraph.core.base.function import Function


class BinaryFunction(Function, ABC):
    def __init__(self,
                 a: Computation,
                 b: Computation):
        super(BinaryFunction, self).__init__(a,b)
        self.a = self.args[0]
        self.b = self.args[1]
        self.a_res = None
        self.b_res = None

    def forward(self) -> np.ndarray:
        self.a_res = self.a.forward()
        self.b_res = self.b.forward()
        return self.apply(a=self.a_res,
                          b=self.b_res)

    def backward(self) -> None:
        if self.a.requires_grad:
            ga = self.gradient_wrt_a(gradient=self.gradient,
                                     a=self.a_res,
                                     b=self.b_res)
            self.a.accumulate(self, ga)
        if self.b.requires_grad:
            gb = self.gradient_wrt_b(gradient=self.gradient,
                                     a=self.a_res,
                                     b=self.b_res)
            self.b.accumulate(self, gb)

    @abstractmethod
    def apply(self,
              a: np.ndarray,
              b: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_wrt_a(self,
                       gradient: np.ndarray,
                       a: np.ndarray,
                       b: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_wrt_b(self,
                       gradient: np.ndarray,
                       a: np.ndarray,
                       b: np.ndarray) -> np.ndarray:
        pass
