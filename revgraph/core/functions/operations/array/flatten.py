import numpy as np

from revgraph.core.functions.base.generic_function import GenericFunction, gradient_wrt_arg


class Flatten(GenericFunction):
    """
    Flattens the entire tensor
    """
    def apply(self,
              xs: np.ndarray,
              exclude_dim_0=True) -> np.ndarray:
        return (np.reshape(xs, (xs.shape[0], -1))
                if exclude_dim_0
                else np.reshape(xs, (-1,)))

    @gradient_wrt_arg(0)
    def d_xs(self, gradient, xs, exclude_dim_0=True):
        return np.reshape(gradient, newshape=xs.shape)
