import numpy as np

from revgraph.core.computations.functions.base.binary_function import BinaryFunction


class TrueDiv(BinaryFunction):
    def apply(self,
              a: np.ndarray,
              b: np.ndarray) -> np.ndarray:
        return a/b

    def gradient_wrt_a(self,
                       gradient: np.ndarray,
                       a: np.ndarray,
                       b: np.ndarray) -> np.ndarray:
        return gradient/b

    def gradient_wrt_b(self,
                       gradient: np.ndarray,
                       a: np.ndarray,
                       b: np.ndarray) -> np.ndarray:
        return -gradient*a / b**2
