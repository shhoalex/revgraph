import numpy as np

from revgraph.core.functions.base.unary_function import UnaryFunction


class Sinh(UnaryFunction):
    def apply(self, a: np.ndarray) -> np.ndarray:
        return np.sinh(a)

    def gradient_wrt_a(self,
                       gradient: np.ndarray,
                       a: np.ndarray) -> np.ndarray:
        return gradient * np.cosh(a)
