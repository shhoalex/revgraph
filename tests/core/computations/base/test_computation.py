import unittest
import numpy as np
from revgraph.core.computations.base.Computation import Computation
from typing import Dict


class ComputationTestCase(unittest.TestCase):
    def test_computation(self):
        class ComputationImpl(Computation):
            def forward(self, feed_dict: Dict[str, 'Computation']):
                return np.array([1])
        self.assertEqual(ComputationImpl().forward({}), np.array([1]))
