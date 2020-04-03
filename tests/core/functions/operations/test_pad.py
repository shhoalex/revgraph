import unittest

import numpy as np

from revgraph.core.values.variable import Variable
from revgraph.core.functions.operations.pad import Pad


class PadTestCase(unittest.TestCase):
    def test_padding_with_scalar_pad_with(self):
        x = Variable([[1,2,3,4,5],
                      [6,7,8,9,10]])
        op = Pad(x, pad_width=1, constant_values=-1)
        expected = np.pad(x.data, pad_width=1, constant_values=-1)
        op.register(op)
        op.new_context()
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        op.accumulate(op, np.ones_like(expected))
        expected = np.ones((2,5))
        actual = x.gradient
        self.assertTrue((expected == actual).all())

    def test_padding_with_1d_pad_width(self):
        x = Variable(np.ones((2,2)))
        op = Pad(x, pad_width=(2,))
        expected = np.pad(x.data, pad_width=(2,))
        op.register(op)
        op.new_context()
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        op.accumulate(op, np.ones_like(expected))
        expected = np.ones((2,2))
        actual = x.gradient
        self.assertTrue((expected == actual).all())

    def test_padding_with_2d_pad_width(self):
        x = Variable(np.ones((2,2)))
        op = Pad(x, pad_width=(1,2))
        expected = np.pad(x.data, pad_width=(1,2))
        op.register(op)
        op.new_context()
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        op.accumulate(op, np.ones_like(expected))
        expected = np.ones((2,2))
        actual = x.gradient
        self.assertTrue((expected == actual).all())

    def test_padding_with_4_pad_width(self):
        x = Variable(np.ones((2,2)))
        op = Pad(x, pad_width=((1,2), (3,4)))
        expected = np.pad(x.data, pad_width=((1,2), (3,4)))
        op.register(op)
        op.new_context()
        actual = op.forward()
        self.assertTrue((expected == actual).all())
        op.accumulate(op, np.ones_like(expected))
        expected = np.ones((2,2))
        actual = x.gradient
        self.assertTrue((expected == actual).all())
