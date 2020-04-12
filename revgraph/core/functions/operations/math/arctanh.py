import numpy as np

from revgraph.core.functions.base.unary_function import UnaryFunction


class ArcTanh(UnaryFunction):
    def apply(self, a: np.ndarray) -> np.ndarray:
        return np.arctanh(a)

    def gradient_wrt_a(self,
                       gradient: np.ndarray,
                       a: np.ndarray) -> np.ndarray:
        return gradient / (1 - a**2)
