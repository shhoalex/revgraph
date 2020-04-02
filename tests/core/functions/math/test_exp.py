import unittest

import numpy as np

from revgraph.core.values.variable import Variable
from revgraph.core.functions.common.exp import Exp


class ExpTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.a = Variable([0,1,2,3])

    def test_exp(self):
        op = Exp(self.a)
        op.register(op)
        op.new_context()
        expected = np.array([np.exp(i) for i in range(4)])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        op.accumulate(op, np.ones_like(expected))
        expected = np.full((4,), 1) * [np.exp(i) for i in range(4)]
        actual = self.a.gradient
        self.assertTrue((expected == actual).all())
