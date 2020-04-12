import unittest

import numpy as np

from revgraph.core.values.variable import Variable
from revgraph.core.functions.operations.math.sub import Sub


class SubTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.a = Variable([[1,2,6], [3,4,5]])
        self.b = Variable([[1,1,1], [1,1,1]])
        self.c = Variable([[1,1,1]])
        self.d = Variable([[1], [2]])
        self.e = Variable(2)

    def test_subtracting_same_shape(self):
        # Forward
        op = Sub(self.a, self.b)
        op.register(op)
        op.new_context()
        expected = np.array([[0,1,5], [2,3,4]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        gradient = np.ones((2,3))
        op.accumulate(op, gradient)
        expected1 = np.ones((2,3))
        expected2 = np.full((2,3), -1)
        actual1 = self.a.gradient
        actual2 = self.b.gradient
        self.assertTrue((expected1 == actual1).all())
        self.assertTrue((expected2 == actual2).all())

    def test_subtracting_matrix_with_different_number_of_rows(self):
        # Forward
        op = Sub(self.a, self.c)
        op.register(op)
        op.new_context()
        expected = np.array([[0,1,5], [2,3,4]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        gradient = np.ones((2,3))
        op.accumulate(op, gradient)
        actual1 = self.a.gradient
        expected2 = np.full((1,3), -2)
        actual2 = self.c.gradient
        self.assertTrue((gradient == actual1).all())
        self.assertTrue((expected2 == actual2).all())

    def test_subtracting_matrix_with_different_number_of_cols(self):
        # Forward
        op = Sub(self.a, self.d)
        op.register(op)
        op.new_context()
        expected = np.array([[0,1,5], [1,2,3]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        gradient = np.ones((2,3))
        op.accumulate(op, gradient)
        expected1 = gradient
        actual1 = self.a.gradient
        expected2 = np.full((2,1), -3)
        actual2 = self.d.gradient
        self.assertTrue((expected1 == actual1).all())
        self.assertTrue((expected2 == actual2).all())

    def test_subtracting_matrix_with_scalar(self):
        # Forward
        op = Sub(self.a, self.e)
        op.register(op)
        op.new_context()
        expected = np.array([[-1,0,4], [1,2,3]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        gradient = np.ones((2,3))
        op.accumulate(op, gradient)
        expected1 = gradient
        actual1 = self.a.gradient
        expected2 = np.array(-6)
        actual2 = self.e.gradient
        self.assertTrue((expected1 == actual1).all())
        self.assertTrue((expected2 == actual2).all())

    def test_subtracting_nested_operations_with_repeated_nodes(self):
        # Forward
        subexp1 = Sub(self.a, self.b)
        subexp2 = Sub(self.b, self.c)
        op = Sub(subexp1, subexp2)
        op.register(op)
        op.new_context()
        expected = np.array([[0,1,5], [2,3,4]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        gradient = np.ones((2,3))
        op.accumulate(op, gradient)
        expected1 = np.ones((2,3))
        expected2 = np.full((2,3), -2)
        expected3 = np.full((1,3), 2)
        self.assertTrue((expected1 == self.a.gradient).all())
        self.assertTrue((expected2 == self.b.gradient).all())
        self.assertTrue((expected3 == self.c.gradient).all())

    def test_subtracting_with_same_node(self):
        # Forward
        op = Sub(Sub(self.a, self.a), Sub(self.a, self.a))
        op.register(op)
        op.new_context()
        expected = np.zeros((2,3))
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        gradient = np.zeros((2,3))
        op.accumulate(op, gradient)
        expected = np.zeros((2,3))
        self.assertTrue((expected == self.a.gradient).all())

