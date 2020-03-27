import unittest

import numpy as np

from revgraph.core.values.variable import Variable
from revgraph.core.functions.operations.pow import Pow


class PowTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.a = Variable([3.,2.,1.,0.,3.,0])
        self.b = Variable([0.,1.,2.,3.,4.,0])

    def test_pow(self):
        op = Pow(self.a, self.b)
        op.register(op)
        op.new_context()
        expected = np.array([1,2,1,0,81,1])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        gradient = np.ones((6,))
        op.accumulate(op, gradient)
        expected1 = np.array([0,1,2,0,108,0])
        expected2 = np.array([0,1.3863,0,0,88.9875,0])
        actual1 = self.a.gradient
        actual2 = self.b.gradient
        self.assertTrue((np.abs(expected1 - actual1) < 0.001).all())
        self.assertTrue((np.abs(expected2 - actual2) < 0.001).all())
