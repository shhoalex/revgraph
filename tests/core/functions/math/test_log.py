import unittest

import numpy as np

from revgraph.core.values.variable import Variable
from revgraph.core.functions.operations.log import Log


class LogTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.a = Variable([1,2,3,4])

    def test_exp(self):
        op = Log(self.a)
        op.register(op)
        op.new_context()
        expected = np.array([np.log(i) for i in range(1,5)])
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        op.accumulate(op, np.ones_like(expected))
        expected = 1/self.a.data
        actual = self.a.gradient
        self.assertTrue((expected == actual).all())
