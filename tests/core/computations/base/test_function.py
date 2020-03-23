import unittest

import numpy as np

from revgraph.core.computations.base.function import Function
from revgraph.core.computations.base.computation import Computation


class FunctionImpl(Function):
    def forward(self):
        return self.args[0]

    def backward(self):
        self.args[0].accumulate(self, self.gradient)


class ComputationImpl(Computation):
    def forward(self):
        return


class FunctionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.a = ComputationImpl(shape=(2,3), requires_grad=True)
        self.f = FunctionImpl(args=[self.a],
                              shape=(2,3),
                              requires_grad=True)
        self.parent = ComputationImpl(shape=(2,3), requires_grad=True)
        self.f.register(self.parent)

    def test_function_propagates_gradient(self):
        self.f.new_context()
        self.assertFalse(self.f.context_completed())
        self.f.accumulate(self.parent, np.ones((2,3)))
        self.assertTrue(self.f.context_completed())
        self.assertEqual(np.ones((2,3)).all(), self.a.gradient.all())

    def test_function_propagates_gradient_after_completion(self):
        # Register twice
        self.f.register(self.parent)
        self.f.new_context()

        # Propagate the first gradient matrix
        self.assertFalse(self.f.context_completed())
        self.f.accumulate(self.parent, np.ones((2,3)))
        self.assertFalse(self.f.context_completed())
        self.assertFalse(self.a.context_completed())
        self.assertIsNone(self.a.gradient)

        # Propagate the second gradient matrix
        x = np.arange(6).reshape(2,3)
        self.f.accumulate(self.parent, x)
        self.assertTrue(self.f.context_completed())
        self.assertTrue(self.a.context_completed())
        self.assertEqual((x+1).all(), self.f.gradient.all())
        self.assertEqual((x+1).all(), self.a.gradient.all())
