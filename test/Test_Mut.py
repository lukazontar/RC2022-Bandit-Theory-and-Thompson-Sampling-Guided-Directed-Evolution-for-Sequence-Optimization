import unittest
import numpy as np

from util.TS_DE.Mut import Mut


class TestMut(unittest.TestCase):
    def test_output_value(self):
        np.random.seed(seed=0)

        x = np.array([0, 1, 0, 1])
        I = [0, 2]
        mu = 0.5
        expected_result = np.array([0, 1, 0, 1])

        result = Mut(x, I, mu)
        self.assertTrue((result == expected_result).all())

    def test_output_type(self):
        np.random.seed(seed=0)

        x = np.array([0, 1, 0, 1])
        I = [0, 2]
        mu = 0.5

        result = Mut(x, I, mu)

        self.assertIsInstance(result, np.ndarray)

    def test_no_targeted_sites(self):
        np.random.seed(seed=0)

        x = np.array([1, 0, 0, 1, 1])
        I = []
        mu = 0.5

        expected_result = x
        result = Mut(x, I, mu)

        self.assertTrue((result == expected_result).all())

    def test_invalid_input(self):
        np.random.seed(seed=0)

        x = np.array([0, 1, 0, 1])
        I = [0, 2]
        mu = 2

        with self.assertRaises(ValueError):
            Mut(x, I, mu)

    def test_no_mutation(self):
        np.random.seed(seed=0)

        x = np.array([0, 1, 0, 1])
        I = [0, 2]
        mu = 0

        expected_result = np.array([0, 1, 0, 1])
        result = Mut(x, I, mu)

        self.assertTrue((result == expected_result).all())

    def test_input_not_binary(self):
        np.random.seed(seed=0)

        x = np.array([1, 0, 2, 0])
        I = []
        mu = 0
        with self.assertRaises(ValueError):
            Mut(x, I, mu)
