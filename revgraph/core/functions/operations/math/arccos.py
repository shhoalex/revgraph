import numpy as np

from revgraph.core.functions.base.unary_function import UnaryFunction


class ArcCos(UnaryFunction):
    def apply(self, a: np.ndarray) -> np.ndarray:
        return np.arccos(a)

    def gradient_wrt_a(self,
                       gradient: np.ndarray,
                       a: np.ndarray) -> np.ndarray:
        return -gradient / np.sqrt(1 - a**2)

