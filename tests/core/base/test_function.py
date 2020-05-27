import unittest

import numpy as np

from revgraph.core.base.function import Function
from revgraph.core.base.tensor import Tensor


class FunctionImpl(Function):
    def forward(self, *args, **kwargs):
        return self.args[0]

    def backward(self, *args, **kwargs):
        self.args[0].accumulate(self, self.gradient)


class TensorImpl(Tensor):
    dependencies = set()

    def forward(self):
        return


class FunctionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.a = TensorImpl(shape=(2, 3), requires_grad=True)
        self.f = FunctionImpl(args=[self.a])
        self.parent = TensorImpl(shape=(2, 3), requires_grad=True)
        self.f.register(self.parent)

    def test_function_propagates_gradient(self):
        self.f.new_context()
        self.assertFalse(self.f.context_completed())
        self.f.accumulate(self.parent, np.ones((2,3)))
        self.assertTrue(self.f.context_completed())
        self.assertTrue((np.ones((2,3)) == self.a.gradient).all())

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
        self.assertTrue(((x+1) == self.f.gradient).all())
        self.assertTrue(((x+1) == self.a.gradient).all())

    def test_gradient_cleared(self):
        self.f.new_context()
        self.f.accumulate(self.parent, np.ones((2,3)))
        self.assertTrue((np.ones((2,3)) == self.a.gradient).all())
        self.assertTrue((np.ones((2,3)) == self.f.gradient).all())
        self.f.clear_gradient()
        self.assertIsNone(self.f.gradient)
        self.assertIsNone(self.a.gradient)


class FunctionKwargTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.a = TensorImpl(shape=(2, 3), requires_grad=True)
        self.f = FunctionImpl(args=[self.a], kwargs={'sum': 3})

    def test_kwarg_exists(self):
        self.assertEqual(self.f.kwargs['sum'], 3)
