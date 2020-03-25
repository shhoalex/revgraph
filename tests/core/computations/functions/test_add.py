import unittest

import numpy as np

from revgraph.core.computations.values.variable import Variable
from revgraph.core.computations.functions.add import Add


class AddTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.a = Variable([[1,2,6], [3,4,5]])
        self.b = Variable([[1,1,1], [1,1,1]])
        self.c = Variable([[1,1,1]])
        self.d = Variable([[1], [2]])
        self.e = Variable(2)

    def test_adding_same_shape(self):
        # Forward
        op = Add(self.a, self.b)
        op.register(op)
        op.new_context()
        expected = np.array([[2,3,7], [4,5,6]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        gradient = np.ones((2,3))
        op.accumulate(op, gradient)
        actual1 = self.a.gradient
        actual2 = self.b.gradient
        self.assertTrue((gradient == actual1).all())
        self.assertTrue((gradient == actual2).all())

    def test_adding_matrix_with_different_number_of_rows(self):
        # Forward
        op = Add(self.a, self.c)
        op.register(op)
        op.new_context()
        expected = np.array([[2,3,7], [4,5,6]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        gradient = np.ones((2,3))
        op.accumulate(op, gradient)
        actual1 = self.a.gradient
        expected2 = np.full((1,3), 2)
        actual2 = self.c.gradient
        self.assertTrue((gradient == actual1).all())
        self.assertTrue((expected2 == actual2).all())

    def test_adding_matrix_with_different_number_of_cols(self):
        # Forward
        op = Add(self.a, self.d)
        op.register(op)
        op.new_context()
        expected = np.array([[2,3,7], [5,6,7]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        gradient = np.ones((2,3))
        op.accumulate(op, gradient)
        expected1 = gradient
        actual1 = self.a.gradient
        expected2 = np.full((2,1), 3)
        actual2 = self.d.gradient
        self.assertTrue((expected1 == actual1).all())
        self.assertTrue((expected2 == actual2).all())

    def test_adding_matrix_with_scalar(self):
        # Forward
        op = Add(self.a, self.e)
        op.register(op)
        op.new_context()
        expected = np.array([[3,4,8], [5,6,7]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        gradient = np.ones((2,3))
        op.accumulate(op, gradient)
        expected1 = gradient
        actual1 = self.a.gradient
        expected2 = np.array(6)
        actual2 = self.e.gradient
        self.assertTrue((expected1 == actual1).all())
        self.assertTrue((expected2 == actual2).all())

    def test_adding_nested_operations(self):
        # Forward
        op = Add(self.a, Add(self.b, self.c))
        op.register(op)
        op.new_context()
        expected = np.array([[3,4,8], [5,6,7]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        gradient = np.ones((2,3))
        op.accumulate(op, gradient)
        expected1 = gradient
        expected2 = gradient
        expected3 = np.full((1,3), 2)
        self.assertTrue((expected1 == self.a.gradient).all())
        self.assertTrue((expected2 == self.b.gradient).all())
        self.assertTrue((expected3 == self.c.gradient).all())

    def test_adding_nested_operations_with_repeated_nodes(self):
        # Forward
        subexp1 = Add(self.a, self.b)
        subexp2 = Add(self.b, self.c)
        op = Add(subexp1, subexp2)
        op.register(op)
        op.new_context()
        expected = np.array([[4,5,9], [6,7,8]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        gradient = np.ones((2,3))
        op.accumulate(op, gradient)
        expected1 = gradient
        expected2 = np.full((2,3), 2)
        expected3 = np.full((1,3), 2)
        self.assertTrue((expected1 == self.a.gradient).all())
        self.assertTrue((expected2 == self.b.gradient).all())
        self.assertTrue((expected3 == self.c.gradient).all())

    def test_adding_with_same_node(self):
        # Forward
        op = Add(Add(self.a, self.a), Add(self.a, self.a))
        op.register(op)
        op.new_context()
        expected = self.a.data * 4
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        # Backward
        gradient = np.ones((2,3))
        op.accumulate(op, gradient)
        expected = np.full((2,3), 4)
        self.assertTrue((expected == self.a.gradient).all())
