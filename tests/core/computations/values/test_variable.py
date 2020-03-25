import unittest

import numpy as np

from revgraph.core.computations.values.variable import Variable


class VariableTestCase(unittest.TestCase):
    def test_variable_is_mutable(self):
        a = Variable(np.zeros((3,3)))
        a.data += 1
        self.assertTrue((a.data == np.ones((3,3))).all())
