from typing import Tuple, Union

import numpy as np

from revgraph.core.functions.base.generic_function import GenericFunction, gradient_wrt_arg


class Pad(GenericFunction):
    def apply(self,
              xs: np.ndarray,
              pad_width: Union[int, Tuple[int, ...]],
              constant_values: Union[int, float] = 0) -> np.ndarray:
        """
        Pad the tensor with constant_values.
        """
        return np.pad(array=xs,
                      pad_width=pad_width,
                      constant_values=constant_values)

    @gradient_wrt_arg(0)
    def d_xs(self, gradient, xs, pad_width, constant_values=0):
        if np.isscalar(pad_width):
            pad_width = [[pad_width, pad_width]]
        elif np.shape(pad_width) == (1,):
            pad_width = [np.concatenate((pad_width, pad_width))]
        elif np.shape(pad_width) == (2,):
            pad_width = [pad_width]
        if np.shape(pad_width)[0] == 1:
            pad_width = np.repeat(pad_width, np.ndim(xs), 0)
        array_filter = tuple(slice(l, -u or None) for l,u in pad_width)
        return gradient[array_filter]
