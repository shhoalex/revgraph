import numpy as np

from revgraph.core.computations.functions.base.binary_function import BinaryFunction


class MatMul(BinaryFunction):
    def apply(self,
              a: np.ndarray,
              b: np.ndarray) -> np.ndarray:
        return a.dot(b)

    def gradient_wrt_a(self,
                       gradient: np.ndarray,
                       a: np.ndarray,
                       b: np.ndarray) -> np.ndarray:
        # https://github.com/mattjj/autodidact/blob/master/autograd/numpy/numpy_vjps.py
        if max(a.ndim, b.ndim) > 2:
            raise NotImplementedError('Does not support derivative of dot product with dimension>2')

        if a.ndim == 0:
            return (b*gradient).sum()
        if a.ndim == 1 and b.ndim == 1:
            return gradient*b
        if a.ndim == 2 and b.ndim == 1:
            return gradient[:, None] * b
        if a.ndim == 1 and b.ndim == 2:
            return b.dot(gradient)
        return gradient.dot(b.T)

    def gradient_wrt_b(self,
                       gradient: np.ndarray,
                       a: np.ndarray,
                       b: np.ndarray) -> np.ndarray:
        # https://github.com/mattjj/autodidact/blob/master/autograd/numpy/numpy_vjps.py
        if max(a.ndim, b.ndim) > 2:
            raise NotImplementedError('Does not support derivative of dot product with dimension>2')

        if b.ndim == 0:
            return (a*gradient).sum()
        if a.ndim == 1 and b.ndim == 1:
            return gradient*a
        if a.ndim == 2 and b.ndim == 1:
            return gradient.dot(a)
        if a.ndim == 1 and b.ndim == 2:
            return a[:, None] * gradient
        return a.T.dot(gradient)
