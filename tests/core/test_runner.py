import unittest

from revgraph.core.values.variable import Variable
from revgraph.core.runner import *


class RunTestCase(unittest.TestCase):
    def test_ignore_invalid_nodes(self):
        a = Variable(1)
        b = Variable(2)
        c = Variable(3)
        d = a+b
        e = b+c
        f = d+e
        expected = 3
        actual = run(d)
        self.assertTrue(expected, actual)

    def test_substitute_placeholder_by_name(self):
        x = Variable([[2,2], [2,2]])
        p = Placeholder(shape=(2,2), name='p')
        expected = x.data
        actual = run(p, {'p': x})
        self.assertTrue((expected == actual).all())

    def test_substitute_placeholder_by_reference(self):
        x = Variable([[2,2], [2,2]])
        p = Placeholder(shape=(2,2), name='p')
        expected = x.data
        actual = run(p, {p: x})
        self.assertTrue((expected == actual).all())

    def test_valid_function_result(self):
        a = Variable(1)
        b = Variable(2)
        c = a+b
        expected = 3
        actual = run(c)
        self.assertIsNotNone(actual)
        self.assertEqual(expected, actual)

    def test_correct_context_assigned(self):
        a = Variable(1)
        b = Variable(2)
        c = a+b
        d = a+1
        new_backward_context({a, b, c})
        self.assertEqual({c}, set(a.ctx))
        self.assertEqual({c}, set(b.ctx))
        self.assertEqual(set(), set(c.ctx))

    def test_repeated_value_accounted(self):
        a = Variable(1)
        e = a*a
        new_backward_context({a, e})
        self.assertEqual(2, a.ctx[e])

    def test_running_multiple_tensors(self):
        a = Variable(3)
        b = Variable(2)
        c = a*b
        ans_a, ans_b, ans_c = run([a, b, c])
        self.assertEqual(ans_a, 3)
        self.assertEqual(ans_b, 2)
        self.assertEqual(ans_c, 6)
