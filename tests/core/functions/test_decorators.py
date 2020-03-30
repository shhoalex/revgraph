import unittest

from revgraph.core.functions.decorators import *


class NoGradDecoTestCase(unittest.TestCase):
    def test_generate_correct_function(self):
        mylen = no_grad(len)
        self.assertTrue(issubclass(mylen, NoGradFunction))
        self.assertEqual(mylen([])(), 0)
        self.assertEqual(mylen([1])(), 1)
