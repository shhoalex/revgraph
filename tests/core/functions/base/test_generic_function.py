import unittest

import numpy as np

from revgraph.core.values.variable import Variable
from revgraph.core.functions.base.generic_function import GenericFunction, gradient_wrt_arg


class GenericFunctionImpl(GenericFunction):
    def apply(self, a, b, d) -> 'np.ndarray':
        return a+b+d

    @gradient_wrt_arg(0)
    def da(self, gradient, *args, **kwargs):
        a,b = args
        return gradient*b + kwargs.get('d')

    @gradient_wrt_arg(1)
    def db(self, gradient, *args, **kwargs):
        return gradient + kwargs.get('d')


class GenericFunctionImpl2(GenericFunction):
    def apply(self, a, b, d) -> 'np.ndarray':
        return a+b+d

    @gradient_wrt_arg(0)
    def da(self, gradient, *args, **kwargs):
        return gradient + 2

    @gradient_wrt_arg(1)
    def db(self, gradient, *args, **kwargs):
        return gradient + 10


class GenericFunctionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.a = Variable(3)
        self.b = Variable(2)
        self.op = GenericFunctionImpl(self.a, self.b, d=1)
        self.op2 = GenericFunctionImpl2(self.a, self.b, d=1)

    def test_forward(self):
        result = self.op.forward()
        self.assertTrue((result == 6).all())

    def test_gradient_wrt_a(self):
        self.op.forward()
        self.op.register(self.op)
        self.op.new_context()
        self.op.accumulate(self.op, 1)
        expected = np.array(3)
        actual = self.a.gradient
        self.assertTrue((expected == actual).all())

    def test_gradient_wrt_b(self):
        self.op.forward()
        self.op.register(self.op)
        self.op.new_context()
        self.op.accumulate(self.op, 1)
        expected = np.array(2)
        actual = self.b.gradient
        self.assertTrue((expected == actual).all())

    def test_multiple_generic_function_impl(self):
        self.op.forward()
        self.op.register(self.op)
        self.op2.forward()
        self.op2.register(self.op2)
        self.op.new_context()
        self.op2.new_context()
        self.op.accumulate(self.op, 1)
        expected = np.array(3)
        actual = self.a.gradient
        self.assertTrue((expected == actual).all())
        expected = np.array(2)
        actual = self.b.gradient
        self.assertTrue((expected == actual).all())
        self.op2.accumulate(self.op2, 1)
        expected = np.array(6)
        actual = self.a.gradient
        self.assertTrue((expected == actual).all())
        expected = np.array(13)
        actual = self.b.gradient
        self.assertTrue((expected == actual).all())

    def test_function_caches_output(self):
        expected = self.op.forward()
        self.assertTrue((expected == self.op.output).all())
