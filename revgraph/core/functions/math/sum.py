from typing import Any

import numpy as np

from revgraph.core.util import repeat_to_match_shape
from revgraph.core.functions.base.generic_function import GenericFunction, gradient_wrt_arg


class Sum(GenericFunction):
    def apply(self,
              xs: np.ndarray,
              axis: Any = None) -> np.ndarray:
        return xs.sum(axis=axis)

    @gradient_wrt_arg(0)
    def dx(self, gradient, xs, axis=None):
        return repeat_to_match_shape(gradient, xs.shape, axis)[0]
