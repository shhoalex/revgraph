import unittest

import revgraph.core as rc

from revgraph.dl.core.layers.dense import dense

from tests.utils import match_structure


class DenseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.placeholder = rc.placeholder(name='x', shape=(None, 10))
        self.builder_basic = dense(units=10, use_bias=False)

    def tearDown(self) -> None:
        del self.placeholder
        del self.builder_basic

    def test_generate_valid_graph(self):
        metadata = self.builder_basic({
            'units': 10,
            'graph': self.placeholder
        })
        expected = self.placeholder.dot(rc.variable([]))
        actual = metadata['graph']
        self.assertTrue(match_structure(expected, actual))
