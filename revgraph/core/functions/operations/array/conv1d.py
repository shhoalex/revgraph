import numpy as np

from revgraph.core.functions.base.generic_function import gradient_wrt_arg
from revgraph.core.functions.operations.array.conv2d import Conv2D


class Conv1D(Conv2D):
    def apply(self, a, b, padding='VALID', stride=1) -> np.ndarray:
        i, j, k = a.shape
        a = a.reshape((i, j, 1, k))
        x, y, z = b.shape
        b = b.reshape((x, 1, y, z))
        ans = super().apply(a, b, padding, stride)
        p, q, _, r = ans.shape
        return ans.reshape((p, q, r))

    @gradient_wrt_arg(0)
    def da(self, gradient, a, b, padding='VALID', stride=1) -> np.ndarray:
        i, j, k = a.shape
        a = a.reshape((i, j, 1, k))
        x, y, z = b.shape
        b = b.reshape((x, 1, y, z))
        p, q, r = gradient.shape
        gradient = gradient.reshape((p, q, 1, r))
        ans = super().da(gradient, a, b, padding, stride)
        return ans.reshape((i, j, k))

    @gradient_wrt_arg(1)
    def db(self, gradient, a, b, padding='VALID', stride=1) -> np.ndarray:
        i, j, k = a.shape
        a = a.reshape((i, j, 1, k))
        x, y, z = b.shape
        b = b.reshape((x, 1, y, z))
        p, q, r = gradient.shape
        gradient = gradient.reshape((p, q, 1, r))
        ans = super().db(gradient, a, b, padding, stride)
        return ans.reshape((x, y, z))
