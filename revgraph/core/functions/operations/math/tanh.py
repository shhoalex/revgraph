import numpy as np

from revgraph.core.functions.base.unary_function import UnaryFunction


class Tanh(UnaryFunction):
    def apply(self, a: np.ndarray) -> np.ndarray:
        return np.tanh(a)

    def gradient_wrt_a(self,
                       gradient: np.ndarray,
                       a: np.ndarray) -> np.ndarray:
        return gradient / np.cosh(a) ** 2
