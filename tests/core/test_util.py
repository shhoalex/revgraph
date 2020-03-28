import unittest

from revgraph.core.util import *


class RepeatToMatchShapeTestCase(unittest.TestCase):
    def test(self):
        a = np.array([[0,1],
                      [1,2],
                      [2,3]])
        b = a.sum()
        expected = np.full((3,2), 9)
        self.assertTrue((expected == repeat_to_match_shape(b, (3,2), None)[0]).all())
