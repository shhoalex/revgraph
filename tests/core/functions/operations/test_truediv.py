import unittest

import numpy as np

from revgraph.core.values.variable import Variable
from revgraph.core.functions.operations.truediv import TrueDiv


class TrueDivTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.a = Variable([[10,12,14], [2,3,4]])
        self.b = Variable([[2,2,1], [2,2,4]])

    def test_truediv(self):
        # Forward
        op = TrueDiv(self.a, self.b)
        op.register(op)
        op.new_context()
        expected = np.array([[5,6,14], [1,1.5,1]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        gradient = np.ones((2,3))
        op.accumulate(op, gradient)
        expected1 = np.array([[0.5,0.5,1], [0.5,0.5,0.25]])
        expected2 = np.array([[-2.5,-3,-14], [-0.5,-0.75,-0.25]])
        actual1 = self.a.gradient
        actual2 = self.b.gradient
        self.assertTrue((expected1 == actual1).all())
        self.assertTrue((expected2 == actual2).all())
