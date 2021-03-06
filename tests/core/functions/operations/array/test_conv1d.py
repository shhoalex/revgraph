import unittest

import numpy as np

from revgraph.core.values.variable import Variable
from revgraph.core.functions.operations.array.conv1d import Conv1D


class Conv1DTestCase(unittest.TestCase):
    def setUp(self):
        self.x = Variable([[[1, 2],
                            [3, 4],
                            [5, 6],
                            [7, 8],
                            [9, 10]],
                           [[11, 12],
                            [13, 14],
                            [15, 16],
                            [17, 18],
                            [19, 20]]])
        self.y = Variable([[[1, 2, 3],
                            [4, 5, 6]],
                           [[7, 8, 9],
                            [10, 11, 12]],
                           [[13, 14, 15],
                            [16, 17, 18]]])

    def test_valid_output_with_no_padding_and_stride_equals_1(self):
        op = Conv1D(self.x, self.y, padding='VALID', stride=1)
        expected = [[[231, 252, 273],
                     [333, 366, 399],
                     [435, 480, 525]],
                    [[741, 822, 903],
                     [843, 936, 1029],
                     [945, 1050, 1155]]]
        actual = op.forward()
        self.assertTrue((expected == actual).all())

    def test_valid_gradient_wrt_x_with_no_padding_and_stride_equals_1(self):
        op = Conv1D(self.x, self.y, padding='VALID', stride=1)
        op.register(op)
        op.new_context()
        expected = [[[6, 15],
                     [30, 48],
                     [72, 99],
                     [66, 84],
                     [42, 51]],
                    [[6, 15],
                     [30, 48],
                     [72, 99],
                     [66, 84],
                     [42, 51]]]
        forward_result = op.forward()
        gradient = np.ones_like(forward_result)
        op.accumulate(op, gradient)
        actual = self.x.gradient
        self.assertTrue((expected == actual).all())

    def test_valid_gradient_wrt_y_with_no_padding_and_stride_equals_1(self):
        op = Conv1D(self.x, self.y, padding='VALID', stride=1)
        op.register(op)
        op.new_context()
        expected = [[[48, 48, 48],
                     [54, 54, 54]],
                    [[60, 60, 60],
                     [66, 66, 66]],
                    [[72, 72, 72],
                     [78, 78, 78]]]
        forward_result = op.forward()
        gradient = np.ones_like(forward_result)
        op.accumulate(op, gradient)
        actual = self.y.gradient
        self.assertTrue((expected == actual).all())
    """
    def test_valid_output_with_zero_padding_and_stride_equals_1(self):
        op = Conv1D(self.x, self.y, padding='SAME', stride=1)
        expected = [[[130,  140,  150],
                     [231,  252,  273],
                     [333,  366,  399],
                     [435,  480,  525],
                     [202,  236,  270]],
                    [[590,  640,  690],
                     [741,  822,  903],
                     [843,  936, 1029],
                     [945, 1050, 1155],
                     [422,  496,  570]]]
        actual = op.forward()
        self.assertTrue((expected == actual).all())

    def test_valid_gradient_wrt_x_with_zero_padding_and_stride_equals_1(self):
        op = Conv1D(self.x, self.y, padding='SAME', stride=1)
        op.register(op)
        op.new_context()
        expected = [[[30, 48],
                     [72, 99],
                     [72, 99],
                     [72, 99],
                     [66, 84]],
                    [[30, 48],
                     [72, 99],
                     [72, 99],
                     [72, 99],
                     [66, 84]]]
        forward_result = op.forward()
        gradient = np.ones_like(forward_result)
        op.accumulate(op, gradient)
        actual = self.x.gradient
        self.assertTrue((expected == actual).all())

    def test_valid_gradient_wrt_y_with_zero_padding_and_stride_equals_1(self):
        op = Conv1D(self.x, self.y, padding='SAME', stride=1)
        op.register(op)
        op.new_context()
        expected = [[[72, 72, 72],
                     [80, 80, 80]],
                    [[100, 100, 100],
                     [110, 110, 110]],
                    [[88, 88, 88],
                     [96, 96, 96]]]
        forward_result = op.forward()
        gradient = np.ones_like(forward_result)
        op.accumulate(op, gradient)
        actual = self.y.gradient
        self.assertTrue((expected == actual).all())
    """
    def test_valid_output_with_no_padding_and_stride_equals_2(self):
        op = Conv1D(self.x, self.y, padding='VALID', stride=2)
        expected = [[[231, 252, 273],
                     [435, 480, 525]],
                    [[741, 822, 903],
                     [945, 1050, 1155]]]
        actual = op.forward()
        self.assertTrue((expected == actual).all())

    def test_valid_gradient_wrt_x_with_no_padding_and_stride_equals_2(self):
        op = Conv1D(self.x, self.y, padding='VALID', stride=2)
        op.register(op)
        op.new_context()
        expected = [[[6, 15],
                     [24, 33],
                     [48, 66],
                     [24, 33],
                     [42, 51]],
                    [[6, 15],
                     [24, 33],
                     [48, 66],
                     [24, 33],
                     [42, 51]]]
        forward_result = op.forward()
        gradient = np.ones_like(forward_result)
        op.accumulate(op, gradient)
        actual = self.x.gradient
        self.assertTrue((expected == actual).all())

    def test_valid_gradient_wrt_y_with_no_padding_and_stride_equals_2(self):
        op = Conv1D(self.x, self.y, padding='VALID', stride=2)
        op.register(op)
        op.new_context()
        expected = [[[32, 32, 32],
                     [36, 36, 36]],
                    [[40, 40, 40],
                     [44, 44, 44]],
                    [[48, 48, 48],
                     [52, 52, 52]]]
        forward_result = op.forward()
        gradient = np.ones_like(forward_result)
        op.accumulate(op, gradient)
        actual = self.y.gradient
        self.assertTrue((expected == actual).all())
    """
    def test_valid_output_with_zero_padding_and_stride_equals_2(self):
        op = Conv1D(self.x, self.y, padding='SAME', stride=2)
        expected = [[[130, 140, 150],
                     [333, 366, 399],
                     [202, 236, 270]],
                    [[590, 640, 690],
                     [843, 936, 1029],
                     [422, 496, 570]]]
        actual = op.forward()
        self.assertTrue((expected == actual).all())

    def test_valid_gradient_wrt_x_with_zero_padding_and_stride_equals_2(self):
        op = Conv1D(self.x, self.y, padding='SAME', stride=2)
        op.register(op)
        op.new_context()
        expected = [[[24, 33],
                     [48, 66],
                     [24, 33],
                     [48, 66],
                     [24, 33]],
                    [[24, 33],
                     [48, 66],
                     [24, 33],
                     [48, 66],
                     [24, 33]]]
        forward_result = op.forward()
        gradient = np.ones_like(forward_result)
        op.accumulate(op, gradient)
        actual = self.x.gradient
        self.assertTrue((expected == actual).all())

    def test_valid_gradient_wrt_y_zero_padding_and_stride_equals_2(self):
        op = Conv1D(self.x, self.y, padding='SAME', stride=2)
        op.register(op)
        op.new_context()
        expected = [[[40, 40, 40],
                     [44, 44, 44]],
                    [[60, 60, 60],
                     [66, 66, 66]],
                    [[40, 40, 40],
                     [44, 44, 44]]]
        forward_result = op.forward()
        gradient = np.ones_like(forward_result)
        op.accumulate(op, gradient)
        actual = self.y.gradient
        self.assertTrue((expected == actual).all())
    """
    def test_previous_gradients_wrt_x_are_accounted(self):
        a = Variable(np.arange(18).reshape(2,3,3))
        op = a * Conv1D(self.x, self.y, padding='VALID', stride=1)
        op.register(op)
        op.new_context()
        forward_result = op.forward()
        expected = [[[0, 252, 546],
                     [999, 1464, 1995],
                     [2610, 3360, 4200]],
                    [[6669, 8220, 9933],
                     [10116, 12168, 14406],
                     [14175, 16800, 19635]]]
        self.assertTrue((expected == forward_result).all())
        gradient = np.ones_like(forward_result.shape)
        op.accumulate(op, gradient)
        expected = [[[8, 17],
                     [52, 97],
                     [186, 294],
                     [340, 439],
                     [296, 359]],
                    [[62, 152],
                     [322, 529],
                     [834, 1185],
                     [934, 1195],
                     [674, 818]]]
        actual = self.x.gradient
        self.assertTrue((expected == actual).all())

    def test_previous_gradients_wrt_y_are_accounted(self):
        a = Variable(np.arange(18).reshape(2,3,3))
        op = a * Conv1D(self.x, self.y, padding='VALID', stride=1)
        op.register(op)
        op.new_context()
        forward_result = op.forward()
        expected = [[[0, 252, 546],
                     [999, 1464, 1995],
                     [2610, 3360, 4200]],
                    [[6669, 8220, 9933],
                     [10116, 12168, 14406],
                     [14175, 16800, 19635]]]
        self.assertTrue((expected == forward_result).all())
        gradient = np.ones_like(forward_result.shape)
        op.accumulate(op, gradient)
        expected = [[[519, 567, 615],
                     [564, 618, 672]],
                    [[609, 669, 729],
                     [654, 720, 786]],
                    [[699, 771, 843],
                     [744, 822, 900]]]
        actual = self.y.gradient
        self.assertTrue((expected == actual).all())
