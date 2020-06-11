import unittest

import revgraph.core as rc

from revgraph.dl.core.layers.dense import dense


class DenseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.placeholder = rc.placeholder(name='x', shape=(None, 10))
        self.build_a = dense(units=10)

    def tearDown(self) -> None:
        del self.placeholder
        del self.build_a

    def test_generate_valid_metadata(self):
        metadata = self.build_a({
            'units': 10,
            'output': self.placeholder
        })
