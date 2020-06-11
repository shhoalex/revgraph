import unittest

import numpy as np

import revgraph.core as rc

from revgraph.dl.core.layers.dense import dense

from tests.utils import match_structure


class DenseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.placeholder = rc.placeholder(name='x', shape=(None, 10))
        self.placeholder_metadata = dict(units=10, graph=self.placeholder)
        self.builder_basic = dense(units=20, use_bias=False)
        self.builder_with_bias = dense(units=30)

    def tearDown(self) -> None:
        del self.placeholder
        del self.placeholder_metadata
        del self.builder_basic
        del self.builder_with_bias

    def test_generate_valid_graph(self):
        metadata = self.builder_basic(self.placeholder_metadata)

        # Metadata keys
        expected = {'graph',
                    'kernel',
                    'units',
                    'use_bias'}
        actual = set(metadata.keys())
        self.assertSetEqual(expected, actual)

        # Graph
        expected = self.placeholder.dot(rc.variable([]))
        actual = metadata['graph']
        self.assertTrue(match_structure(expected, actual))

        # Kernel
        kernel = metadata['kernel']
        self.assertTrue(isinstance(kernel.data, np.ndarray))
        self.assertEqual((10, 20), kernel.shape)

        # Units
        units = metadata['units']
        self.assertEqual(20, units)

        # Use bias
        use_bias = metadata['use_bias']
        self.assertFalse(use_bias)

    def test_builder_with_bias_generates_valid_graph(self):
        metadata = self.builder_with_bias(self.placeholder_metadata)

        # Expected keys
        expected = {'bias',
                    'graph',
                    'kernel',
                    'units',
                    'use_bias'}
        actual = set(metadata.keys())
        self.assertSetEqual(expected, actual)

        # Graph
        expected = self.placeholder.dot(rc.variable([])) + rc.variable([])
        actual = metadata['graph']
        self.assertTrue(match_structure(expected, actual))

        # Kernel
        kernel = metadata['kernel']
        self.assertTrue(isinstance(kernel.data, np.ndarray))
        self.assertEqual((10, 30), kernel.shape)

        # Bias
        bias = metadata['bias']
        self.assertTrue(isinstance(bias.data, np.ndarray))
        self.assertEqual((1, 30), bias.shape)

        # Units
        units = metadata['units']
        self.assertEqual(30, units)

        # Use bias
        use_bias = metadata['use_bias']
        self.assertTrue(use_bias)
