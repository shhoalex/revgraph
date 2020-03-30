import unittest

from revgraph.core.values.variable import Variable
from revgraph.core.functions.base.no_grad_function import NoGradFunction


class NoGradFunctionImpl(NoGradFunction):
    def call(x):
        return 2*x


two = NoGradFunctionImpl


class NoGradFunctionTestCase(unittest.TestCase):
    def test_output_is_valid(self):
        x = Variable(1)
        y = x+two(x)
        self.assertEqual(y(), 3)

    def test_no_gradient_is_propagated(self):
        x = Variable(1)
        y = x + two(x)
        y.register(y)
        y.new_context()
        y.accumulate(y, 1)
        self.assertEqual(x.gradient, 1)

    def test_raises_exception_when_invalid_argument_is_propagated(self):
        with self.assertRaises(TypeError):
            two()()
        with self.assertRaises(TypeError):
            two(1,2)()

    def test_output_is_valid_when_using_kwarg(self):
        node = two(x=3)
        self.assertEqual(node(), 6)
