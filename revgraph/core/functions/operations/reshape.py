from typing import Tuple

import numpy as np

from revgraph.core.functions.base.generic_function import GenericFunction, gradient_wrt_arg


class Reshape(GenericFunction):
    def apply(self,
              xs: np.ndarray,
              newshape: Tuple[int, ...],
              order=None) -> np.ndarray:
        return np.reshape(xs, newshape)

    @gradient_wrt_arg(0)
    def d_xs(self, gradient, xs, newshape, order=None):
        return np.reshape(gradient, newshape=xs.shape, order=order)
