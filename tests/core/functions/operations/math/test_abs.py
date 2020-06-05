import unittest

import numpy as np

from revgraph.core.values.variable import Variable
from revgraph.core.functions.operations.math.abs import Abs


class AbsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.a = Variable([[-1,2,-3], [-4,5,-6]])
        self.b = Variable([[0,-1,2,-3], [0,-4,5,-6]])

    def test_absolute(self):
        # Forward
        op = Abs(self.a)
        op.register(self.a)
        op.new_context()
        expected = np.array([[1,2,3], [4,5,6]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        op.accumulate(self.a, np.ones((1,3)))
        expected = np.array([[-1,1,-1], [-1,1,-1]])
        actual = self.a.gradient
        self.assertTrue((expected == actual).all())

    def test_absolute_with_zero(self):
        # Forward
        op = Abs(self.b)
        op.register(self.b)
        op.new_context()
        expected = np.array([[0,1,2,3], [0,4,5,6]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        op.accumulate(self.b, np.ones((2,4)))
        expected = np.array([[0,-1,1,-1], [0,-1,1,-1]])
        actual = self.b.gradient
        self.assertTrue((expected == actual).all())
