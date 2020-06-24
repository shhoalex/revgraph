import unittest

import numpy as np

import revgraph.core as rc

from revgraph.dl.core.layers.reshape import reshape

from tests.utils import match_structure


class ReshapeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.x = rc.constant(data=np.arange(8).reshape(2, 4))
        self.metadata_x = {
            'units': (2, 4),
            'graph': self.x
        }
        self.builder = reshape(new_shape=(8,))

    def tearDown(self) -> None:
        del self.x
        del self.metadata_x

    def test_generate_valid_graph(self):
        metadata = self.builder(self.metadata_x)
        expected = {'graph',
                    'units',
                    'regularized_nodes'}
        actual = set(metadata.keys())
        self.assertSetEqual(expected, actual)

        # Graph
        expected = rc.reshape(rc.constant([]), ())
        actual = metadata['graph']
        self.assertTrue(match_structure(expected, actual))
