import unittest

import numpy as np

from revgraph.dl.session.train_test_validation_split import train_test_validation_split


class TrainTestValidationSplitTestCase(unittest.TestCase):
    def test_valid_input(self):
        x = np.array([[i*j for j in range(5)] for i in range(10)])
        y = np.array([[i*j for j in range(2)] for i in range(10)])
        (x0, y0), (x1, y1), (x2, y2) = train_test_validation_split(
            x=x,
            y=y,
            train=0.7,
            test=0.2,
            validation=0.1
        )
        eq = lambda a, b: (a.shape == b.shape) and (a == b).all()
        self.assertTrue(eq(x0, x[:7]))
        self.assertTrue(eq(y0, y[:7]))
        self.assertTrue(eq(x1, x[7:9]))
        self.assertTrue(eq(y1, y[7:9]))
        self.assertTrue(eq(x2, x[9:10]))
        self.assertTrue(eq(y2, y[9:10]))

        (x0, y0), (x1, y1), (x2, y2) = train_test_validation_split(
            x=x,
            y=y,
            train=1.0,
            test=0.0,
            validation=0.0
        )
        eq = lambda a, b: (a.shape == b.shape) and (a == b).all()
        self.assertTrue(eq(x0, x[:10]))
        self.assertTrue(eq(y0, y[:10]))
        self.assertTrue(eq(x1, x[:0]))
        self.assertTrue(eq(y1, y[:0]))
        self.assertTrue(eq(x2, x[:0]))
        self.assertTrue(eq(y2, y[:0]))

        (x0, y0), (x1, y1), (x2, y2) = train_test_validation_split(
            x=x,
            y=y,
            train=0.8,
            test=0.2,
            validation=0.0
        )
        eq = lambda a, b: (a.shape == b.shape) and (a == b).all()
        self.assertTrue(eq(x0, x[:8]))
        self.assertTrue(eq(y0, y[:8]))
        self.assertTrue(eq(x1, x[8:10]))
        self.assertTrue(eq(y1, y[8:10]))
        self.assertTrue(eq(x2, x[:0]))
        self.assertTrue(eq(y2, y[:0]))

        (x0, y0), (x1, y1), (x2, y2) = train_test_validation_split(
            x=x,
            y=y,
            train=0.8,
            test=0.0,
            validation=0.2
        )
        eq = lambda a, b: (a.shape == b.shape) and (a == b).all()
        self.assertTrue(eq(x0, x[:8]))
        self.assertTrue(eq(y0, y[:8]))
        self.assertTrue(eq(x1, x[:0]))
        self.assertTrue(eq(y1, y[:0]))
        self.assertTrue(eq(x2, x[8:10]))
        self.assertTrue(eq(y2, y[8:10]))
