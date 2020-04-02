import numpy as np

from revgraph.core.functions.base.generic_function import GenericFunction


class Gradient(GenericFunction):
    def forward(self):
        super().forward(self.args[0])
        self.backward(self.args[0])
        return self.data

    def backward(self, node):
        node.accumulate(self, self.data)

    def apply(self, a):
        return np.ones_like(a)
