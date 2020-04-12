import unittest

import numpy as np

from revgraph.core.values.variable import Variable
from revgraph.core.functions.operations.math.min import Min


class MinTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.a = Variable([[0,1,2],
                           [-2,4,4],
                           [1,9,5],
                           [2,4,-2]])

    def test_correct_return_value(self):
        op = Min(self.a)
        op.register(op)
        op.new_context()
        expected = -2
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        op.accumulate(op, np.ones_like(expected))
        expected = np.array([[0,0,0],
                             [0.5,0,0],
                             [0,0,0],
                             [0,0,0.5]])
        actual = self.a.gradient
        self.assertTrue((expected == actual).all())

    def test_correct_return_value_with_axis_equals_0(self):
        op = Min(self.a, axis=0)
        op.register(op)
        op.new_context()
        expected = [-2,1,-2]
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        op.accumulate(op, np.ones_like(expected))
        expected = np.array([[0,1,0],
                             [1,0,0],
                             [0,0,0],
                             [0,0,1]])
        actual = self.a.gradient
        self.assertTrue((expected == actual).all())

    def test_correct_return_value_with_axis_equals_1(self):
        op = Min(self.a, axis=1)
        op.register(op)
        op.new_context()
        expected = [0,-2,1,-2]
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        op.accumulate(op, np.ones_like(expected))
        expected = np.array([[1,0,0],
                             [1,0,0],
                             [1,0,0],
                             [0,0,1]])
        actual = self.a.gradient
        self.assertTrue((expected == actual).all())

    def test_correct_return_value_with_keepdims(self):
        op = Min(self.a, keepdims=True)
        op.register(op)
        op.new_context()
        expected = [[-2]]
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        op.accumulate(op, np.ones_like(expected))
        expected = np.array([[0,0,0],
                             [0.5,0,0],
                             [0,0,0],
                             [0,0,0.5]])
        actual = self.a.gradient
        self.assertTrue((expected == actual).all())

    def test_correct_return_value_with_axis_equals_0_and_keepdims(self):
        op = Min(self.a, axis=0, keepdims=True)
        op.register(op)
        op.new_context()
        expected = [[-2,1,-2]]
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        op.accumulate(op, np.ones_like(expected))
        expected = np.array([[0,1,0],
                             [1,0,0],
                             [0,0,0],
                             [0,0,1]])
        actual = self.a.gradient
        self.assertTrue((expected == actual).all())

    def test_correct_return_value_with_axis_equals_1_and_keepdims(self):
        op = Min(self.a, axis=1, keepdims=True)
        op.register(op)
        op.new_context()
        expected = [[0],
                    [-2],
                    [1],
                    [-2]]
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        op.accumulate(op, np.ones_like(expected))
        expected = np.array([[1,0,0],
                             [1,0,0],
                             [1,0,0],
                             [0,0,1]])
        actual = self.a.gradient
        self.assertTrue((expected == actual).all())
