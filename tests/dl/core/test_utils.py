import unittest

import revgraph.dl.core.utils as utils


class ValidateTestCase(unittest.TestCase):
    def test_throws_error_when_invalid_predicate_is_passed(self):
        with self.assertRaises(ValueError) as err:
            utils.validate(
                (False, 'An error'),
                (True, 'No error')
            )
            self.assertEqual(err.exception.args, 'An error')

        with self.assertRaises(ValueError) as err:
            utils.validate(
                (True, 'No error0'),
                (False, 'An error1'),
                (True, 'No error1')
            )
            self.assertEqual(err.exception.args, 'An error1')

    def test_does_not_throw_error_when_valid_predicate_is_passed(self):
        utils.validate()
        utils.validate(
            (True, 'No error0'),
            (True, 'No error1')
        )
