import unittest

import numpy as np

from revgraph.core.computations.values.placeholder import Placeholder


class PlaceholderTestCase(unittest.TestCase):
    def setUp(self):
        self.p = Placeholder(shape=(3,2),
                             requires_grad=False,
                             default=[[0,1],[1,0],[1,1]],
                             name='p')
        self.q = Placeholder(shape=(None,2),
                             requires_grad=False)

    def test_name_is_set(self):
        self.assertEqual(self.p.name, 'p')

    def test_default_is_used(self):
        expected = np.array([[0,1],
                             [1,0],
                             [1,1]]).all()
        self.assertEqual(expected, self.p.feed().data.all())

    def test_value_is_used(self):
        expected = np.array([[1,1],
                             [1,1],
                             [1,1]])
        self.assertEqual(expected.all(), self.p.feed(expected).data.all())

    def test_valid_shape(self):
        a = np.ones((3,4))
        self.assertFalse(self.p.valid_shape(a))

        a = np.ones((3,2))
        self.assertTrue(self.p.valid_shape(a))

        a = np.ones((2,3,1))
        self.assertFalse(self.p.valid_shape(a))

        a = np.ones((1,2,3))
        self.assertFalse(self.p.valid_shape(a))

    def test_valid_shape_with_none(self):
        a = np.ones((1,2,2))
        self.assertFalse(self.q.valid_shape(a))

        a = np.ones((1,2,2))
        self.assertFalse(self.q.valid_shape(a))

        a = np.ones((3,2))
        self.assertTrue(self.q.valid_shape(a))

        a = np.ones((1,2))
        self.assertTrue(self.q.valid_shape(a))
