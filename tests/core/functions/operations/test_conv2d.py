import unittest

import numpy as np

from revgraph.core.values.variable import Variable
from revgraph.core.functions.operations.conv2d import Conv2D


class Conv2DTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.x = Variable(np.arange(120).reshape(5, 4, 3, 2))
        self.y = Variable(np.arange(60).reshape(2, 3, 5, 2))

    def test_forward_output_with_no_padding_and_stride_equals_1(self):
        op = Conv2D(self.x, self.y, padding='VALID', stride=1)
        actual = op.forward()
        self.assertEqual(actual.shape, (2, 2, 2, 2))
        expected = [[[[20410, 20920],
                      [24760, 25420]],
                     [[37810, 38920],
                      [42160, 43420]]],
                    [[[72610, 74920],
                      [76960, 79420]],
                     [[90010, 92920],
                      [94360, 97420]]]]
        self.assertTrue((expected == actual).all())

    def test_gradient_wrt_x_with_no_padding_and_stride_equals_1(self):
        da = [[[[1.0, 5.0, 9.0, 13.0, 17.0],
                [22.0, 30.0, 38.0, 46.0, 54.0],
                [62.0, 70.0, 78.0, 86.0, 94.0],
                [41.0, 45.0, 49.0, 53.0, 57.0]],
               [[62.0, 70.0, 78.0, 86.0, 94.0],
                [164.0, 180.0, 196.0, 212.0, 228.0],
                [244.0, 260.0, 276.0, 292.0, 308.0],
                [142.0, 150.0, 158.0, 166.0, 174.0]],
               [[61.0, 65.0, 69.0, 73.0, 77.0],
                [142.0, 150.0, 158.0, 166.0, 174.0],
                [182.0, 190.0, 198.0, 206.0, 214.0],
                [101.0, 105.0, 109.0, 113.0, 117.0]]],
              [[[1.0, 5.0, 9.0, 13.0, 17.0],
                [22.0, 30.0, 38.0, 46.0, 54.0],
                [62.0, 70.0, 78.0, 86.0, 94.0],
                [41.0, 45.0, 49.0, 53.0, 57.0]],
               [[62.0, 70.0, 78.0, 86.0, 94.0],
                [164.0, 180.0, 196.0, 212.0, 228.0],
                [244.0, 260.0, 276.0, 292.0, 308.0],
                [142.0, 150.0, 158.0, 166.0, 174.0]],
               [[61.0, 65.0, 69.0, 73.0, 77.0],
                [142.0, 150.0, 158.0, 166.0, 174.0],
                [182.0, 190.0, 198.0, 206.0, 214.0],
                [101.0, 105.0, 109.0, 113.0, 117.0]]]]
        op = Conv2D(self.x, self.y, padding='VALID', stride=1)
        op.register(op)
        op.new_context()
        result = op.forward()
        gradient = np.ones_like(result)
        op.accumulate(op, gradient)
        actual = self.x.gradient
        self.assertTrue((da == actual).all())

    def test_gradient_wrt_y_with_no_padding_and_stride_equals_1(self):
        db = [[[[340.0, 340.0], [348.0, 348.0], [356.0, 356.0], [364.0, 364.0], [372.0, 372.0]],
               [[380.0, 380.0], [388.0, 388.0], [396.0, 396.0], [404.0, 404.0], [412.0, 412.0]],
               [[420.0, 420.0], [428.0, 428.0], [436.0, 436.0], [444.0, 444.0], [452.0, 452.0]]],
              [[[500.0, 500.0], [508.0, 508.0], [516.0, 516.0], [524.0, 524.0], [532.0, 532.0]],
               [[540.0, 540.0], [548.0, 548.0], [556.0, 556.0], [564.0, 564.0], [572.0, 572.0]],
               [[580.0, 580.0], [588.0, 588.0], [596.0, 596.0], [604.0, 604.0], [612.0, 612.0]]]]
        op = Conv2D(self.x, self.y, padding='VALID', stride=1)
        op.register(op)
        op.new_context()
        result = op.forward()
        gradient = np.ones_like(result)
        op.accumulate(op, gradient)
        actual = self.y.gradient
        self.assertTrue((db == actual).all())
