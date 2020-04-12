import unittest

from revgraph.core.values.variable import Variable
from revgraph.core.runner import *
"""

class RunnerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.x = Variable([[0,0],
                           [0,1]])
        self.p = Placeholder(shape=(2,2), name='p')
        self.expected = np.array([[0,0], [0,1]])

    def test_correct_return_value(self):
        runner = Runner(node=self.x)
        actual = runner.run()
        self.assertTrue((self.expected == actual).all())

    def test_substitute_placeholder_by_reference(self):
        runner = Runner(node=self.p)
        actual = runner.run({self.p: self.x})
        self.assertTrue((self.expected == actual).all())

    def test_substitute_placeholder_by_name(self):
        runner = Runner(node=self.p)
        actual = runner.run({'p': self.x})
        self.assertTrue((self.expected == actual).all())

"""
class RunTestCase(unittest.TestCase):
    def test_ignore_invalid_nodes(self):
        a = Variable(1)
        b = Variable(2)
        c = Variable(3)
        d = a+b
        e = b+c
        f = d+e
        expected = 3
        [actual] = run(d)
        self.assertTrue(expected, actual)

    def test_substitute_placeholder_by_name(self):
        x = Variable([[2,2], [2,2]])
        p = Placeholder(shape=(2,2), name='p')
        expected = x.data
        [actual] = run(p, {'p': x})
        self.assertTrue((expected == actual).all())

    def test_substitute_placeholder_by_reference(self):
        x = Variable([[2,2], [2,2]])
        p = Placeholder(shape=(2,2), name='p')
        expected = x.data
        [actual] = run(p, {p: x})
        self.assertTrue((expected == actual).all())

    def test_valid_function_result(self):
        a = Variable(1)
        b = Variable(2)
        c = a+b
        expected = 3
        [actual] = run(c)
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
