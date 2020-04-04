import numpy as np

from revgraph.core.functions.base.generic_function import GenericFunction, gradient_wrt_arg


def convolve(a, b):
    n_a, *_ = a.shape
    n_b, *_ = b.shape


class Conv1D(GenericFunction):
    def apply(self, a, b, stride=1, padding='VALID'):
        return a

    @gradient_wrt_arg(0)
    def da(self, gradient, a, b, stride=1, padding='VALID'):
        return gradient

    @gradient_wrt_arg(1)
    def db(self, gradient, a, b, stride=1, padding='VALID'):
        return gradient
