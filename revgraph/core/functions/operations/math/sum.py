from typing import Any

import numpy as np

from revgraph.core.functions.base.generic_function import GenericFunction, gradient_wrt_arg


class Sum(GenericFunction):
    def apply(self,
              xs: np.ndarray,
              axis: Any = None,
              keepdims: bool = False) -> np.ndarray:
        return xs.sum(axis=axis, keepdims=keepdims)

    @gradient_wrt_arg(0)
    def dx(self, gradient, xs, axis=None, keepdims=False):
        return self.match_shape(gradient, xs.shape, axis)
