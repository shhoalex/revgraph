import unittest

import numpy as np

from revgraph.core.values.variable import Variable
from revgraph.core.functions.operations.reshape import Reshape


class ReshapeTestCase(unittest.TestCase):
    def test_gradient_reshaped(self):
        x = Variable([[1,2,3,4,5],
                      [6,7,8,9,10]])
        op = Reshape(x, (5,2))
        op.register(op)
        op.new_context()
        expected = (np.arange(10) + 1).reshape(5,2)
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        op.accumulate(op, np.ones((5,2)))
        expected = np.ones_like(x.data)
        actual = x.gradient
        self.assertTrue((expected == actual).all())
