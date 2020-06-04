from typing import Any

import numpy as np

from revgraph.core.functions.base.generic_function import GenericFunction, gradient_wrt_arg


class Min(GenericFunction):
    def apply(self,
              xs: np.ndarray,
              axis: Any = None,
              keepdims: bool = False) -> np.ndarray:
        return np.min(xs, axis=axis, keepdims=keepdims)

    @gradient_wrt_arg(0)
    def d_xs(self, gradient, xs, axis=None, keepdims=False):
        gradients = self.match_shape(gradient, xs.shape, axis)
        mask = xs == self.match_shape(self.data, xs.shape, axis)
        n = np.sum(mask, axis=axis, keepdims=True)
        return gradients * mask / n
