import numpy as np

from revgraph.core.functions.base.unary_function import UnaryFunction


class Sqrt(UnaryFunction):
    def apply(self, a: np.ndarray) -> np.ndarray:
        return np.sqrt(a)

    def gradient_wrt_a(self,
                       gradient: np.ndarray,
                       a: np.ndarray) -> np.ndarray:
        return gradient * 0.5 * a**-0.5
