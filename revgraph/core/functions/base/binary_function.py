from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from revgraph.core.base.tensor import Tensor
from revgraph.core.base.function import Function


class BinaryFunction(Function, ABC):
    """
    A function that takes 2 tensors and accumulates gradient on both of them.
    """
    def __init__(self,
                 a: Union[Tensor, list, int, float],
                 b: Union[Tensor, list, int, float]):
        super(BinaryFunction, self).__init__(args=[a,b])
        self.a = self.args[0]
        self.b = self.args[1]
        self.a_res = None
        self.b_res = None

    def forward(self, *args, **kwargs) -> np.ndarray:
        self.a_res = self.a.forward()
        self.b_res = self.b.forward()
        self.data = self.apply(a=self.a_res,
                               b=self.b_res)
        return self.data

    def backward(self, *args, **kwargs) -> None:
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
