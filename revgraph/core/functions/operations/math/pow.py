import numpy as np

from revgraph.core.functions.base.binary_function import BinaryFunction


class Pow(BinaryFunction):
    def apply(self,
              a: np.ndarray,
              b: np.ndarray) -> np.ndarray:
        return a**b

    def gradient_wrt_a(self,
                       gradient: np.ndarray,
                       a: np.ndarray,
                       b: np.ndarray) -> np.ndarray:
        # d(a^b)/da = if b==0 then 0 else b*a^(b-1)
        return gradient * b * a ** np.where(b==0, 0, b-1)

    def gradient_wrt_b(self,
                       gradient: np.ndarray,
                       a: np.ndarray,
                       b: np.ndarray) -> np.ndarray:
        # if b=0 then gradient=0 since a^0=1.
        # np.where(a,a,1): temporarily mask 0 with 1 to avoid computing log(0)
        return gradient * np.where(np.logical_or(a==0, b==0),
                                   0,
                                   np.log(np.where(a,a,1)) * a**b)
