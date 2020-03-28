from typing import Any

import numpy as np

from revgraph.core.util import repeat_to_match_shape
from revgraph.core.functions.base.generic_function import GenericFunction, gradient_wrt_arg


class Sum(GenericFunction):
    def apply(self,
              *args: np.ndarray,
              **kwargs: Any) -> np.ndarray:
        axis = kwargs.pop('axis', None)
        xs = args[0]
        return xs.sum(axis=axis)

    @gradient_wrt_arg(0)
    def dx(self, gradient, *args, **kwargs):
        xs = args[0]
        shape = xs.shape
        return repeat_to_match_shape(gradient, shape, kwargs.pop('axis', None))[0]
