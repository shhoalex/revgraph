import unittest

import numpy as np

from revgraph.core.computations.base.Function import Function
from revgraph.core.computations.base.Value import Value


class FunctionImpl(Function):
    def forward(self):
        return self.args[0]

    def backward(self):
        self.args[0].accumulate(self.gradient)


class FunctionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.a = Value(data=np.ones((2,3)), shape=(2,3), requires_grad=True)
        self.f = FunctionImpl(args=[self.a],
                              shape=(2,3),
                              requires_grad=True)
        self.parent = Value(shape=(2,3), requires_grad=True)
        self.f.register(self.parent)

    def test_function_propagates_gradient(self):
        self.f.new_context()
        self.assertFalse(self.f.context_completed())
        self.f.accumulate(self.parent, np.ones((2,3)))
        self.assertTrue(self.f.context_completed())
        self.assertEqual(np.ones((2,3)).all(), self.a.gradient.all())
