import unittest

from revgraph.core.values.variable import Variable
from revgraph.core.functions.miscellaneous import *


class LenTestCase(unittest.TestCase):
    def test_len_does_not_affect_gradient(self):
        x = Variable([[1,2,3],
                      [4,5,6],
                      [7,8,9]])
        op = x * len(x)
        self.assertTrue((op.forward() == [[3,6,9],
                                          [12,15,18],
                                          [21,24,27]]).all())
        op.register(op)
        op.new_context()
        op.accumulate(op, [[1,1,1],
                           [1,1,1],
                           [1,1,1]])
        self.assertTrue((x.gradient == [[3,3,3],
                                        [3,3,3],
                                        [3,3,3]]).all())
