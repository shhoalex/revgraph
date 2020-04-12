import unittest

from revgraph.core.values.variable import Variable
from revgraph.core.values.placeholder import Placeholder
from revgraph.core.controls.simple_loop import SimpleLoop
from revgraph.core.functions.decorators import no_grad


@no_grad
def inc(x):
    x += 2


class SimpleLoopTestCase(unittest.TestCase):
    def test_looping(self):
        x = Variable(0)
        SimpleLoop(10, inc(x))()
        self.assertEqual(20, x.data.item())

    def test_feed_dict_with_iteration(self):
        xs = []

        @no_grad
        def push(i):
            xs.append(i.item())

        c = Placeholder(name='c', shape=())
        SimpleLoop(4, push(c), lambda i: {
            'c': i
        })()
        self.assertEqual([0, 1, 2, 3], xs)

    def test_feed_dict(self):
        xs = []

        @no_grad
        def push(i):
            xs.append(i.item())

        c = Placeholder(name='c', shape=())
        SimpleLoop(4, push(c), {
            'c': 1
        })()
        self.assertEqual([1, 1, 1, 1], xs)
