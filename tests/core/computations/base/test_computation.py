import unittest
import numpy as np
from revgraph.core.computations.base.Computation import Computation


class ComputationTestCase(unittest.TestCase):
    def test_computation(self):
        class ComputationImpl(Computation):
            def forward(self):
                return np.array([1])
        self.assertEqual(ComputationImpl().forward(), np.array([1]))
