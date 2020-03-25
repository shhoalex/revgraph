import unittest

import numpy as np

from revgraph.core.computations.base.computation import Computation


class ComputationImpl(Computation):
    def forward(self):
        return


class ComputationGradientTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.a = ComputationImpl((3,4), True)
        cls.b = ComputationImpl((3,4), True)

    def setUp(self) -> None:
        self.c = ComputationImpl(None, True)
        self.c.register(self.a)
        self.c.register(self.b)
        self.c.new_context()

    def tearDown(self) -> None:
        del self.c

    def test_gradient_successfully_created(self):
        self.c.accumulate(self.a, np.ones((3,4)))
        self.assertEqual(self.c.shape, (3,4))
        self.assertTrue((self.c.gradient == np.ones((3,4))).all())

    def test_gradient_successfully_accumulated(self):
        self.c.accumulate(self.a, np.ones((3,4)))
        self.c.accumulate(self.b, np.ones((3,4)))
        result = np.full((3,4), 2)
        self.assertTrue((self.c.gradient == result).all())

    def test_gradient_rejected_for_repeated_parent_node(self):
        self.c.accumulate(self.a, np.ones((3,4)))
        with self.assertRaises(ValueError):
            self.c.accumulate(self.a, np.ones((3,4)))

    def test_gradient_rejected_for_invalid_parent_node(self):
        self.c.accumulate(self.a, np.ones((3,4)))
        self.assertEqual(self.c.ctx[self.a], 0)
        with self.assertRaises(ValueError):
            self.c.accumulate(self.c, np.ones((3,4)))

    def test_context_completed_correctly(self):
        self.c.accumulate(self.a, np.ones((3,4)))
        self.assertFalse(self.c.context_completed())
        self.c.accumulate(self.b, np.ones((3,4)))
        self.assertTrue(self.c.context_completed())

    def test_context_completed_correctly_for_repeated_parent_node(self):
        self.c.register(self.a)
        self.c.new_context()
        self.c.accumulate(self.a, np.ones((3,4)))
        self.assertFalse(self.c.context_completed())
        self.c.accumulate(self.a, np.ones((3, 4)))
        self.assertFalse(self.c.context_completed())
        self.c.accumulate(self.b, np.ones((3,4)))
        self.assertTrue(self.c.context_completed())


class ComputationUnbroadcastingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.a = ComputationImpl((1,5), True)
        self.b = ComputationImpl((5,1), True)
        self.c = ComputationImpl((3,2), True)

    def tearDown(self) -> None:
        del self.a, self.b, self.c

    def test_unbroadcast_with_same_shape(self):
        m = np.ones((1,5))
        result = self.a.unbroadcast(m)
        self.assertEqual(result.shape, (1,5))
        self.assertTrue((result == np.array([[1,1,1,1,1]])).all())

        m = np.ones((5,1))
        result = self.b.unbroadcast(m)
        self.assertEqual(result.shape, (5,1))
        self.assertTrue((result == np.array([[1, 1, 1, 1, 1]]).T).all())

    def test_unbroadcast_with_different_row(self):
        m = np.ones((2,5))
        result = self.a.unbroadcast(m)
        self.assertEqual(result.shape, (1,5))
        self.assertTrue((result == np.array([[2,2,2,2,2]])).all())

        m = np.ones((10,5))
        result = self.a.unbroadcast(m)
        self.assertEqual(result.shape, (1,5))
        self.assertTrue((result == np.array([[10,10,10,10,10]])).all())

        m = np.ones((1,1))
        result = self.b.unbroadcast(m)
        self.assertEqual(result.shape, (5,1))
        self.assertTrue((result == np.array([[1, 1, 1, 1, 1]]).T).all())

    def test_unbroadcast_with_different_col(self):
        m = np.ones((1,1))
        result = self.a.unbroadcast(m)
        self.assertEqual(result.shape, (1,5))
        self.assertTrue((result == np.array([[1,1,1,1,1]])).all())

        m = np.ones((5,5))
        result = self.b.unbroadcast(m)
        self.assertEqual(result.shape, (5,1))
        self.assertTrue((result == np.array([[5,5,5,5,5]]).T).all())

    def test_unbroadcast_with_extra_dimension(self):
        m = np.ones((5,))
        result = self.a.unbroadcast(m)
        self.assertEqual(result.shape, (1,5))
        self.assertTrue((result == np.array([[1,1,1,1,1]])).all())

    def test_unbroadcast_with_different_col_and_row(self):
        m = np.ones((1,))
        result = self.c.unbroadcast(m)
        self.assertEqual(result.shape, (3,2))
        self.assertTrue((result == np.ones((3,2))).all())

    def test_unbroadcast_with_invalid_shape(self):
        m = np.ones((3,3))
        with self.assertRaises(ValueError):
            self.c.unbroadcast(m)

        m = np.ones((4,2))
        with self.assertRaises(ValueError):
            self.c.unbroadcast(m)
