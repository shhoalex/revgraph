import unittest

import numpy as np

from revgraph.core.computations.values.variable import Variable
from revgraph.core.computations.values.placeholder import Placeholder
from revgraph.core.computations.runner import Runner


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
