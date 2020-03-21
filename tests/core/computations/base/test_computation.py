import unittest

import numpy as np

from revgraph.core.computations.base.Computation import Computation


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
        self.c.new_session()

    def tearDown(self) -> None:
        del self.c

    def test_gradient_successfully_created(self):
        self.c.accumulate(self.a, np.ones((3,4)))
        self.assertEqual(self.c.shape, (3,4))
        self.assertEqual(self.c.gradient.all(), np.ones((3,4)).all())

    def test_gradient_successfully_accumulated(self):
        self.c.accumulate(self.a, np.ones((3,4)))
        self.c.accumulate(self.b, np.ones((3,4)))
        result = np.array((3,4))
        result.fill(2)
        self.assertEqual(self.c.gradient.all(), result.all())

    def test_gradient_rejected_for_repeated_parent_node(self):
        self.c.accumulate(self.a, np.ones((3,4)))
        with self.assertRaises(ValueError) as error:
            self.assertEqual(error.exception.args[0], 'Invalid node for gradient propagation')
            self.c.accumulate(self.a, np.ones((3,4)))

    def test_gradient_rejected_for_invalid_parent_node(self):
        self.c.accumulate(self.a, np.ones((3,4)))
        self.assertEqual(self.c.ctx[self.a], 0)
        with self.assertRaises(ValueError) as error:
            self.assertEqual(error.exception.args[0], 'Invalid node for gradient propagation')
            self.c.accumulate(self.c, np.ones((3,4)))
