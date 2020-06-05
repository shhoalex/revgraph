import numpy as np

from revgraph.core.functions.base.unary_function import UnaryFunction


class Abs(UnaryFunction):
    def apply(self, a: np.ndarray) -> np.ndarray:
        return np.abs(a)

    def gradient_wrt_a(self,
                       gradient: np.ndarray,
                       a: np.ndarray) -> np.ndarray:
        # d|x|/dx = x/|x|
        return gradient * np.sign(a)
