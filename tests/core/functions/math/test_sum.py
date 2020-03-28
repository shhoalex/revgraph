import unittest

import numpy as np

from revgraph.core.values.variable import Variable
from revgraph.core.functions.math.sum import Sum


class SumTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.a = Variable([[0,1],
                           [1,0],
                           [1,1],
                           [2,4]])

    def test_sum_across_axis_0(self):
        op = Sum(self.a, axis=0)
        op.register(op)
        op.new_context()
        expected = np.array([4,6])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        op.accumulate(op, np.ones_like(expected))
        expected = np.ones((4,2))
        actual = self.a.gradient
        self.assertTrue((expected == actual).all())

    def test_sum_across_axis_1(self):
        op = Sum(self.a, axis=1)
        op.register(op)
        op.new_context()
        expected = np.array([1,1,2,6])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        op.accumulate(op, np.ones_like(expected))
        expected = np.ones((4,2))
        actual = self.a.gradient
        self.assertTrue((expected == actual).all())

    def test_sum_across_all_axes(self):
        op = Sum(self.a)
        op.register(op)
        op.new_context()
        expected = np.array([10])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        op.accumulate(op, np.ones_like(expected))
        expected = np.ones((4,2))
        actual = self.a.gradient
        self.assertTrue((expected == actual).all())
