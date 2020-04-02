import numpy as np

from revgraph.core.functions.base.unary_function import UnaryFunction


class ArcSinh(UnaryFunction):
    def apply(self, a: np.ndarray) -> np.ndarray:
        return np.arcsinh(a)

    def gradient_wrt_a(self,
                       gradient: np.ndarray,
                       a: np.ndarray) -> np.ndarray:
        return gradient / np.sqrt(a**2 + 1)
