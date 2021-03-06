import unittest

import numpy as np

from revgraph.core.values.variable import Variable
from revgraph.core.functions.base.binary_function import BinaryFunction


class BinaryFunctionImpl(BinaryFunction):
    def apply(self,
              a: np.ndarray,
              b: np.ndarray) -> np.ndarray:
        return a+b*2

    def gradient_wrt_a(self,
                       gradient: np.ndarray,
                       a: np.ndarray,
                       b: np.ndarray) -> np.ndarray:
        return gradient

    def gradient_wrt_b(self,
                       gradient: np.ndarray,
                       a: np.ndarray,
                       b: np.ndarray) -> np.ndarray:
        return gradient*2


class BinaryFunctionTestCase(unittest.TestCase):
    def setUp(self):
        self.a = Variable([[1,2], [2,3]])
        self.b = Variable([[4,5], [6,7]])
        self.f = BinaryFunctionImpl(a=self.a,
                                    b=self.b)
        self.f.register(self.f)
        self.f.new_context()

    def test_correct_value_forwarded(self):
        expected = np.array([[9,12], [14,17]])
        actual = self.f.forward()
        self.assertTrue((expected == actual).all())

    def test_correct_gradient_propagated_wrt_first_arg(self):
        gradient = np.ones((2,2))
        expected = np.ones((2,2))
        self.f.accumulate(self.f, gradient)
        actual = self.a.gradient
        self.assertTrue((expected == actual).all())

    def test_correct_gradient_propagated_wrt_second_arg(self):
        gradient = np.ones((2,2))
        expected = np.full((2,2), 2)
        self.f.accumulate(self.f, gradient)
        actual = self.b.gradient
        self.assertTrue((expected == actual).all())

    def test_function_caches_output(self):
        expected = self.f.forward()
        self.assertTrue((expected == self.f.data).all())

    def test_correct_dependencies(self):
        a = Variable(0)
        b = Variable(1)
        c = a+b
        d = Variable(2)
        e = b+d
        f = c+d
        self.assertEqual({a}, a.dependencies)
        self.assertEqual({b}, b.dependencies)
        self.assertEqual({c,a,b}, c.dependencies)
        self.assertEqual({d}, d.dependencies)
        self.assertEqual({e,b,d}, e.dependencies)
        self.assertEqual({f,c,d,a,b}, f.dependencies)
