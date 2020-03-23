import unittest

from revgraph.core.computations.base.computation import Computation
from revgraph.core.computations.values.constant import Constant


class ComputationImpl(Computation):
    def forward(self):
        pass


class ConstantTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.a = Constant([[1,0,0],
                           [0,1,0]])

    def test_shape_is_correct(self):
        self.assertEqual((2,3), self.a.shape)

    def test_data_is_immutable(self):
        with self.assertRaises(ValueError):
            self.a.data += 1

    def test_register_parent_node_causes_error(self):
        b = ComputationImpl((2,3), True)
        with self.assertRaises(ValueError):
            self.a.register(b)

    def test_context_completed(self):
        self.assertTrue(self.a.context_completed())

    def test_accumulate_causes_error(self):
        with self.assertRaises(ValueError):
            self.a.accumulate(self.a, [[1,1,1],
                                       [1,1,1]])
