import numpy as np

from revgraph.core.functions.base.generic_function import GenericFunction, gradient_wrt_arg


def padded(a, padding='VALID'):
    if padding is 'VALID':
        return a
    elif padding is 'SAME':
        zero_pad = [[0] * a.shape[2]]
        return np.array([np.concatenate((zero_pad, a_, zero_pad)) for a_ in a])
    else:
        raise ValueError(f'Invalid padding option: {padding}')


def conv1d(a, b, stride=1, padding='VALID'):
    a = padded(a, padding)
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


def da(gradient, a, b, stride=1, padding='VALID'):
    n_a, in_col, in_channel = a.shape
    n_b, in_channel_, out_channel = b.shape
    ans = np.zeros_like(a)

    for n in range(n_a):
        for i in range(in_col):
            for j in range(in_channel):
                d = in_col - n_b
                x = gradient[n, max(0,i-d):i+1, :]
                y = b[max(0,i-d):i+1, j, :][::-1]
                ans[n,i,j] = (x*y).sum()
    return ans


def db(gradient, a, b, stride=1, padding='VALID'):
    ans = np.zeros_like(b)
    n_a, in_col, in_channel = a.shape
    n_b, in_channel_, out_channel = b.shape
    for n in range(n_b):
        for i in range(in_channel):
            for j in range(out_channel):
                x = gradient[:, :, j].flatten()
                y = a[:, n:n+out_channel, i].flatten()
                ans[n,i,j] = x@y
    return ans


class Conv1D(GenericFunction):
    def apply(self, a, b, stride=1, padding='VALID'):
        return conv1d(a, b, stride, padding)

    @gradient_wrt_arg(0)
    def da(self, gradient, a, b, stride=1, padding='VALID'):
        return da(gradient, a, b, stride, padding)

    @gradient_wrt_arg(1)
    def db(self, gradient, a, b, stride=1, padding='VALID'):
        return db(gradient, a, b, stride, padding)
