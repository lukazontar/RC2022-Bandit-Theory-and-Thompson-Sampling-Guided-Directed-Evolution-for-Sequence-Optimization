import unittest
import numpy as np

from util.Common.Rcb import Rcb


class TestRcb(unittest.TestCase):
    def test_output_value(self):
        np.random.seed(seed=0)

        x = np.array([1, 0, 1, 0])
        y = np.array([0, 1, 0, 1])

        result = Rcb(x, y)
        expected_result = np.array([0, 1, 0, 1])

        self.assertTrue((np.sum(result == x) + np.sum(result == y)) == len(result))
        self.assertTrue(np.array_equal(result, expected_result))

    def test_output_type(self):
        np.random.seed(seed=0)

        x = np.array([1, 0, 1, 0])
        y = np.array([0, 1, 0, 1])

        result = Rcb(x, y)

        self.assertIsInstance(result, np.ndarray)

    def test_input_different_length(self):
        np.random.seed(seed=0)

        x = np.array([1, 0, 1, 0])
        y = np.array([0, 1, 0, 1, 0])

        with self.assertRaises(ValueError):
            Rcb(x, y)

    def test_input_not_binary(self):
        np.random.seed(seed=0)

        x = np.array([1, 0, 2, 0])
        y = np.array([0, 1, 0, 1])

        with self.assertRaises(ValueError):
            Rcb(x, y)
