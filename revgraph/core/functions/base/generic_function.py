from abc import ABC, abstractmethod
from typing import Union, Any, Callable
from inspect import getmembers

import numpy as np


from revgraph.core.base.tensor import Tensor
from revgraph.core.base.function import Function


GradientFunction = Callable[
    ['GenericFunction', np.ndarray, np.ndarray, Any],
    np.ndarray
]


def gradient_wrt_arg(key: Any) -> GradientFunction:
    def decorator(function):
        function.gradient_wrt = key
        return function
    return decorator


class GenericFunction(Function, ABC):
    """
    A function that takes n argument(s) may not be tensor and accumulates
    gradient on all the argument (same bug as the one in tensor.py).
    This might be the most buggy/hacky class in the entire project.
    """
    def __init__(self,
                 *args: Union[Tensor, list, int, float],
                 **kwargs: Any):
        super(GenericFunction, self).__init__(args=args,
                                              kwargs=kwargs)
        self.results = None

        # Hacky way to bind gradient function (for calculating
        # derivative wrt arg[i]) with its respective argument

        self.gradient_wrt_arg = {}
        for _,m in getmembers(self):
            if hasattr(m, 'gradient_wrt'):
                self.gradient_wrt_arg[m.gradient_wrt] = m

    def forward(self, *args, **kwargs) -> np.ndarray:
        self.results = [arg.forward() for arg in self.args]
        self.data = self.apply(*self.results, **self.kwargs)
        return self.data

    def backward(self, *args, **kwargs) -> None:
        for i,arg in enumerate(self.args):
            if arg.requires_grad:
                g_arg = self.gradient_wrt_arg[i](
                    self.gradient,
                    *self.results,
                    **self.kwargs
                )
                arg.accumulate(self, g_arg)

    @abstractmethod
    def apply(self,
              *args: np.ndarray,
              **kwargs: Any) -> np.ndarray:
        pass
