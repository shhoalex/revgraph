import unittest

from revgraph.core.values.variable import Variable
from revgraph.core.functions.base.no_grad_function import NoGradFunction


def two(x):
    f = NoGradFunction(x)
    f.target_function = lambda x: 2*x
    return f


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

    def test_correct_dependencies(self):
        a = Variable([1,2,3])
        b = two(a)
        x = a+b
        y = two(x)
        z = Variable(3)
        c = z+x
        self.assertEqual({a}, a.dependencies)
        self.assertEqual({b,a}, b.dependencies)
        self.assertEqual({x,a,b}, x.dependencies)
        self.assertEqual({y,x,a,b}, y.dependencies)
        self.assertEqual({z}, z.dependencies)
        self.assertEqual({c,z,x,a,b}, c.dependencies)
