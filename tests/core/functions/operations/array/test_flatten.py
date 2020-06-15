import unittest

import numpy as np

from revgraph.core.values.variable import Variable
from revgraph.core.functions.operations.array.flatten import Flatten


class FlattenTestCase(unittest.TestCase):
    def test_gradient_flattened_excluding_dim_0(self):
        x = Variable([[[1, 2], [3, 4], [5, 6]],
                      [[7, 8], [9, 10], [11, 12]],
                      [[13, 14], [15, 16], [17, 18]]])
        op = Flatten(x)
        op.register(op)
        op.new_context()
        expected = (np.arange(18) + 1).reshape(3, 6)
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        op.accumulate(op, (np.arange(18) + 1).reshape(3, 6))
        expected = (np.arange(18) + 1).reshape(3, 3, 2)
        actual = x.gradient
        self.assertTrue((expected == actual).all())

    def test_gradient_flattened_including_dim_0(self):
        x = Variable([[[1, 2], [3, 4], [5, 6]],
                      [[7, 8], [9, 10], [11, 12]],
                      [[13, 14], [15, 16], [17, 18]]])
        op = Flatten(x, exclude_dim_0=False)
        op.register(op)
        op.new_context()
        expected = (np.arange(18) + 1).reshape(18,)
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        op.accumulate(op, (np.arange(18) + 1).reshape(18,))
        expected = (np.arange(18) + 1).reshape(3, 3, 2)
        actual = x.gradient
        self.assertTrue((expected == actual).all())
