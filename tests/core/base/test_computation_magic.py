import unittest

from revgraph.core.values.variable import Variable
from revgraph.core.values.placeholder import Placeholder


class ComputationMagicTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.a = Variable(1)
        self.b = Variable(2)
        self.c = 3

    def test_add(self):
        self.assertEqual((self.a+self.b)(), 3)
        self.assertEqual((self.c+self.a)(), 4)

    def test_sub(self):
        self.assertEqual((self.a-self.b)(), -1)
        self.assertEqual((self.c-self.a)(), 2)

    def test_mul(self):
        self.assertEqual((self.a*self.b)(), 2)
        self.assertEqual((self.c*self.a)(), 3)

    def test_floordiv(self):
        self.assertEqual((self.a//self.b)(), 0)
        self.assertEqual((self.c//self.a)(), 3)

    def test_truediv(self):
        self.assertEqual((self.a/self.b)(), 0.5)
        self.assertEqual((self.c/self.a)(), 3.0)

    def test_pow(self):
        self.assertEqual((self.a**self.b)(), 1)
        self.assertEqual((self.c**self.a)(), 3)

    def test_matmul(self):
        self.assertEqual(Variable([1]).matmul(Variable([2]))(), [2])
        self.assertEqual(Variable([1]).dot(Variable([2]))(), [2])
        self.assertEqual((Variable([1]) @ Variable([2]))(), [2])

    def test_call(self):
        x = Placeholder(shape=(), name='x')
        a = x(x=3)
        self.assertEqual(a, 3)
