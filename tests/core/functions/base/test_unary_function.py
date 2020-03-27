import unittest

import numpy as np

from revgraph.core.values.variable import Variable
from revgraph.core.functions.base.unary_function import UnaryFunction


class UnaryFunctionImpl(UnaryFunction):
    def apply(self, a: np.ndarray) -> np.ndarray:
        return a*2

    def gradient_wrt_a(self,
                       gradient: np.ndarray,
                       a: np.ndarray) -> np.ndarray:
        return gradient*2


class UnaryFunctionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.x = Variable([[1,2], [2,3]])
        self.double = UnaryFunctionImpl(a=self.x)
        self.double.register(self.double)
        self.double.new_context()

    def test_correct_value_forwarded(self):
        expected = np.array([[2,4], [4,6]])
        self.assertTrue((expected == self.double.forward()).all())

    def test_correct_gradient_propagated(self):
        gradient = np.ones((2,2))
        expected = np.full((2,2), 2)
        self.double.accumulate(self.double, gradient)
        actual = self.x.gradient
        self.assertTrue((expected == actual).all())
