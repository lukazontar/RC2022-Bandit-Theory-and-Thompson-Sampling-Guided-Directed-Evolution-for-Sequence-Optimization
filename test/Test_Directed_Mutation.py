import unittest
import numpy as np

from util.TS_DE.Directed_Mutation import Directed_Mutation


class TestDirectedMutation(unittest.TestCase):

    def test_output_value(self):
        np.random.seed(seed=0)

        d = 20
        S = [np.random.randint(0, 2, d) for _ in range(10)]
        theta = np.random.rand(d)
        mu = 0.1

        result = Directed_Mutation(d=d, S=S, theta=theta, mu=mu)
        expected_result = [np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]),
                           np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0]),
                           np.array([0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1]),
                           np.array([1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0]),
                           np.array([0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0]),
                           np.array([1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0]),
                           np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0]),
                           np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1]),
                           np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]),
                           np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0])]

        self.assertTrue(np.array_equal(result, expected_result))

    def test_output_type(self):
        np.random.seed(seed=0)

        d = 20
        S = [np.random.randint(0, 2, d) for _ in range(10)]
        theta = np.random.rand(d)
        mu = 0.1

        result = Directed_Mutation(d=d, S=S, theta=theta, mu=mu)

        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], np.ndarray)

    def test_output_size(self):
        np.random.seed(seed=0)

        d = 20
        S = [np.random.randint(0, 2, d) for _ in range(10)]
        theta = np.random.rand(d)
        mu = 0.1

        S_ = Directed_Mutation(d=d, S=S, theta=theta, mu=mu)

        result = len(S_)
        expected_result = len(S)

        self.assertEqual(result, expected_result)
