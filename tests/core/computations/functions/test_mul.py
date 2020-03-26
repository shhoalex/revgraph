import unittest

import numpy as np

from revgraph.core.computations.values.variable import Variable
from revgraph.core.computations.functions.mul import Mul


class MulTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.a = Variable([[1,2,6], [3,4,5]])
        self.b = Variable([[2,2,2], [1,1,2]])
        self.c = Variable([[1,2,3]])
        self.d = Variable([[1], [2]])
        self.e = Variable(2)

    def test_multiplying_same_shape(self):
        # Forward
        op = Mul(self.a, self.b)
        op.register(op)
        op.new_context()
        expected = np.array([[2,4,12], [3,4,10]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        gradient = np.ones((2,3))
        op.accumulate(op, gradient)
        expected1 = np.array([[2,2,2], [1,1,2]])
        expected2 = np.array([[1,2,6], [3,4,5]])
        actual1 = self.a.gradient
        actual2 = self.b.gradient
        self.assertTrue((expected1 == actual1).all())
        self.assertTrue((expected2 == actual2).all())

    def test_multiplying_matrix_with_different_number_of_rows(self):
        # Forward
        op = Mul(self.a, self.c)
        op.register(op)
        op.new_context()
        expected = np.array([[1,4,18], [3,8,15]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        gradient = np.ones((2,3))
        op.accumulate(op, gradient)
        expected1 = np.array([[1,2,3], [1,2,3]])
        actual1 = self.a.gradient
        expected2 = np.array([[4,6,11]])
        actual2 = self.c.gradient
        self.assertTrue((expected1 == actual1).all())
        self.assertTrue((expected2 == actual2).all())

    def test_multiplying_matrix_with_different_number_of_cols(self):
        # Forward
        op = Mul(self.a, self.d)
        op.register(op)
        op.new_context()
        expected = np.array([[1,2,6], [6,8,10]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        gradient = np.ones((2,3))
        op.accumulate(op, gradient)
        expected1 = np.array([[1,1,1], [2,2,2]])
        actual1 = self.a.gradient
        expected2 = np.array([[9], [12]])
        actual2 = self.d.gradient
        self.assertTrue((expected1 == actual1).all())
        self.assertTrue((expected2 == actual2).all())

    def test_multiplying_matrix_with_scalar(self):
        # Forward
        op = Mul(self.a, self.e)
        op.register(op)
        op.new_context()
        expected = np.array([[2,4,12], [6,8,10]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        gradient = np.ones((2,3))
        op.accumulate(op, gradient)
        expected1 = np.array([[2,2,2], [2,2,2]])
        actual1 = self.a.gradient
        expected2 = np.array(21)
        actual2 = self.e.gradient
        self.assertTrue((expected1 == actual1).all())
        self.assertTrue((expected2 == actual2).all())
