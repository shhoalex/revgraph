import numpy as np

from .base.unary_function import UnaryFunction


class Neg(UnaryFunction):
    def apply(self, a: np.ndarray) -> np.ndarray:
        return -a

    def gradient_wrt_a(self,
                       gradient: np.ndarray,
                       a: np.ndarray) -> np.ndarray:
        return -gradient
