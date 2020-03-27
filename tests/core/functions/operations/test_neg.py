import unittest

import numpy as np

from revgraph.core.values.variable import Variable
from revgraph.core.functions.operations.neg import Neg


class NegTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.a = Variable([[1,2,3], [4,5,6]])

    def test_negation(self):
        # Forward
        op = Neg(self.a)
        op.register(op)
        op.new_context()
        expected = np.array([[-1,-2,-3], [-4,-5,-6]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        op.accumulate(op, np.ones((1,3)))
        expected = np.array([[-1,-1,-1], [-1,-1,-1]])
        actual = self.a.gradient
        self.assertTrue((expected == actual).all())

    def test_double_negation(self):
        # Forward
        op = Neg(Neg(self.a))
        op.register(op)
        op.new_context()
        expected = np.array([[1,2,3], [4,5,6]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        op.accumulate(op, np.ones((1,3)))
        expected = np.ones((1,3))
        actual = self.a.gradient
        self.assertTrue((expected == actual).all())
