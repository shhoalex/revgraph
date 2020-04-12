import unittest

import numpy as np

from revgraph.core.values.variable import Variable
from revgraph.core.functions.operations.math.matmul import MatMul


class MatMulTestCase(unittest.TestCase):
    def test_dot_product_of_matrix_with_same_shape(self):
        a = Variable([[1,2,3],
                      [4,5,6],
                      [7,8,9]])
        b = Variable([[2,3,4],
                      [5,6,7],
                      [8,9,0]])
        op = MatMul(a,b)
        op.register(op)
        op.new_context()
        expected = np.array([[36,42,18],
                             [81,96,51],
                             [126,150,84]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        gradient = np.ones((3,3))
        op.accumulate(op, gradient)
        expected1 = np.array([[9,18,17],
                              [9,18,17],
                              [9,18,17]])
        expected2 = np.array([[12,12,12],
                              [15,15,15],
                              [18,18,18]])
        actual1 = a.gradient
        actual2 = b.gradient
        self.assertTrue((expected1 == actual1).all())
        self.assertTrue((expected2 == actual2).all())

    def test_dot_product_of_matrix_with_different_row_and_col(self):
        a = Variable([[1,2,3]])
        b = Variable([[4],
                      [5],
                      [6]])
        op = MatMul(a,b)
        op.register(op)
        op.new_context()
        expected = np.array([[32]])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        gradient = np.ones((1,1))
        op.accumulate(op, gradient)
        expected1 = np.array([[4,5,6]])
        expected2 = np.array([[1],
                              [2],
                              [3]])
        actual1 = a.gradient
        actual2 = b.gradient
        self.assertTrue((expected1 == actual1).all())
        self.assertTrue((expected2 == actual2).all())

    def test_dot_product_of_matrix_with_vector(self):
        a = Variable([[1,2],
                      [3,4],
                      [5,6]])
        b = Variable([2,3])
        op = MatMul(a,b)
        op.register(op)
        op.new_context()
        expected = np.array([8,18,28])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        gradient = np.ones((3,))
        op.accumulate(op, gradient)
        expected1 = np.array([[2,3],
                              [2,3],
                              [2,3]])
        expected2 = np.array([9,12])
        actual1 = a.gradient
        actual2 = b.gradient
        self.assertTrue((expected1 == actual1).all())
        self.assertTrue((expected2 == actual2).all())
