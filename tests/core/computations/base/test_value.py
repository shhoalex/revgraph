import unittest

import numpy as np

from revgraph.core.computations.base.Value import Value


class ValueTestCase(unittest.TestCase):
    def test_valid_shape(self):
        a = Value([[1,2,3],
                   [4,5,6]])
        self.assertEqual(a.shape, (2,3))

        b = Value([])
        self.assertEqual(b.shape, (0,))

    def test_accept_valid_data(self):
        a = Value(shape=(2,3))
        data = [[1,1,1],
                [1,1,1]]
        a.data = data
        self.assertEqual((2,3), a.data.shape)

        a = Value(shape=(None,None))
        data = [[1,1,1], [1,1,1]]
        a.data = data
        self.assertEqual((2,3), a.data.shape)

    def test_reject_invalid_data(self):
        a = Value(shape=(2,3))
        data = [[1,2,3],[1,2,3],[1,2,3]]
        with self.assertRaises(ValueError):
            a.data = data

        b = Value(shape=(None,2,2))
        data = [[1,2],[1,2]]
        with self.assertRaises(ValueError):
            a.data = data

        c = Value(shape=(2,2,None))
        with self.assertRaises(ValueError):
            a.data = data

    def test_forward_correct_data(self):
        a = Value([[1,2,3],
                   [4,5,6]], shape=(2,3))
        expected = np.array([[1,2,3],
                             [4,5,6]]).all()
        self.assertEqual(expected, a.forward().all())
