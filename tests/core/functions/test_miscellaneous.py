import unittest

import numpy as np

from revgraph.core.values.variable import Variable
from revgraph.core.functions.miscellaneous import *


class LenTestCase(unittest.TestCase):
    def test_len_does_not_affect_gradient(self):
        x = Variable([[1,2,3],
                      [4,5,6],
                      [7,8,9]])
        op = x * len(x)
        self.assertTrue((op.forward() == [[3,6,9],
                                          [12,15,18],
                                          [21,24,27]]).all())
        op.register(op)
        op.new_context()
        op.accumulate(op, [[1,1,1],
                           [1,1,1],
                           [1,1,1]])
        self.assertTrue((x.gradient == [[3,3,3],
                                        [3,3,3],
                                        [3,3,3]]).all())

    def test_correct_dependencies(self):
        x = Variable([[1,2,3],
                      [4,5,6],
                      [7,8,9]])
        b = len(x)
        c = x*b
        d = b+x
        self.assertEqual({x}, x.dependencies)
        self.assertEqual({b,x}, b.dependencies)
        self.assertEqual({c,b,x}, c.dependencies)
        self.assertEqual({d,b,x}, d.dependencies)


class ComparisonTestCase(unittest.TestCase):
    def test_comparison_returns_boolean_matrix(self):
        x = Variable([[1,2,3],
                      [4,5,6],
                      [7,8,9]])
        y = x>=5
        expected = np.array([[False]*3,
                             [False] + [True]*2,
                             [True]*3])
        actual = y()
        self.assertTrue((expected == actual).all())

    def test_comparison_mix_with_gradient_operator(self):
        x = Variable([[1,2,3],
                      [4,5,6],
                      [7,8,9]])
        y = (x>5) * 2
        expected = np.array([[0,0,0],
                             [0,0,2],
                             [2,2,2]])
        actual = y()
        self.assertTrue((expected == actual).all())

    def test_correct_gradient_propagated(self):
        x = Variable([[1,2],
                      [3,4]])
        y = (x>2) * x
        expected = np.array([[0,0],
                             [3,4]])
        y.register(y)
        actual = y()
        self.assertTrue((expected == actual).all())
        y.accumulate(y, np.ones_like(actual))
        expected = np.array([[0,0], [1,1]])
        actual = x.gradient
        self.assertTrue((expected == actual).all())

    def test_any(self):
        x = Variable([[1,2,3],
                      [4,5,6]])
        y = (x>5).any()
        self.assertTrue(y())

        x = Variable([[1,2,3], [4,5,6]])
        y = (x>6).any()
        self.assertFalse(y())

    def test_all(self):
        x = Variable([[1,2,3]])
        y = (x>0).all()
        self.assertTrue(y())

        x = Variable([[1,2,3]])
        y = (x>1).all()
        self.assertFalse(y())
