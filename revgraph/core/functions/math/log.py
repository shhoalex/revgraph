import numpy as np

from revgraph.core.functions.base.unary_function import UnaryFunction


class Log(UnaryFunction):
    def apply(self, a: np.ndarray) -> np.ndarray:
        return np.log(a)

    def gradient_wrt_a(self,
                       gradient: np.ndarray,
                       a: np.ndarray) -> np.ndarray:
        return gradient / a
