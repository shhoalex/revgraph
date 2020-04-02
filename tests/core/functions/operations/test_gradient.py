import unittest

from revgraph.core.functions.operations.gradient import Gradient
from revgraph.core.values.variable import Variable


class GradientTestCase(unittest.TestCase):
    def test_gradient_is_propagated(self):
        x = Variable(1)
        y = Gradient(x)
        y.new_context()
        y.forward()
        self.assertTrue(x.gradient == 1)
