import numpy as np

from revgraph.core.functions.base.generic_function import GenericFunction, gradient_wrt_arg


def conv1d(a, b, stride=1):
    n_a, in_col, in_channel = a.shape
    n_b, in_channel_, out_channel = b.shape
    assert in_channel == in_channel_, f'in_channel must be equal'
    r = ((in_col - n_b) // stride) + 1
    ans = np.zeros((n_a, r, out_channel))

    for i in range(0, r):
        for f in range(in_channel):
            i_ = i*stride
            ans[:, i, :] += a[:, i_:i_+n_b, f] @ b[:, f, :]
    return ans


class Conv1D(GenericFunction):
    def apply(self, a, b, stride=1, padding='VALID'):
        return conv1d(a,b,stride)

    @gradient_wrt_arg(0)
    def da(self, gradient, a, b, stride=1, padding='VALID'):
        return gradient

    @gradient_wrt_arg(1)
    def db(self, gradient, a, b, stride=1, padding='VALID'):
        return gradient
