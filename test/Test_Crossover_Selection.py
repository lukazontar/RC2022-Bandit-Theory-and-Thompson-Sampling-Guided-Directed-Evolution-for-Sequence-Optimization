import unittest
import numpy as np

from util.common import linear_bayesian_utility_model
from util.TS_DE.BayesRGT import BayesRGT
from util.TS_DE.Crossover_Selection import Crossover_Selection


class TestCrossoverSelection(unittest.TestCase):
    def test_output_value(self):
        np.random.seed(seed=0)

        S = [np.array([0, 1, 1, 0]), np.array([1, 0, 0, 1])]
        theta = np.array([0.2, 0.1, 0.3, 0.4])
        f = lambda x, theta: sum(x * theta)

        expected_result = [np.array([1, 1, 1, 1]), np.array([1, 1, 1, 0])]
        result = Crossover_Selection(S, theta, f)

        self.assertTrue(np.array_equal(result, expected_result))

    def test_output_type(self):
        np.random.seed(seed=0)

        S = [np.array([0, 1, 0]), np.array([1, 0, 1])]
        theta = np.array([0.5, 0.5, 0.5])
        f = lambda x, theta: sum(x * theta)

        result = Crossover_Selection(S, theta, f)

        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], np.ndarray)

    def test_output_size(self):
        np.random.seed(seed=0)

        S = [np.array([0, 1, 0]), np.array([1, 0, 1]), np.array([1, 1, 0])]
        theta = np.array([0.5, 0.5, 0.5])
        f = lambda x, theta: sum(x * theta)

        S_ = Crossover_Selection(S, theta, f)

        result = len(S_)
        expected_result = len(S)

        self.assertEqual(result, expected_result)

    def test_output_BayesRGT(self):
        np.random.seed(seed=0)

        S = [np.array([0, 1, 0]), np.array([1, 0, 1]), np.array([1, 1, 0])]
        d = 3
        theta = np.array([0.5, 0.5, 0.5])
        f = lambda x, theta: sum(x * theta)

        S_ = Crossover_Selection(S, theta, f)
        theta_star = np.random.multivariate_normal(np.zeros(d), np.eye(d))

        prev_result = BayesRGT(M=3,
                               T=1,
                               f=linear_bayesian_utility_model,
                               theta_star=theta_star,
                               thetas=[theta],
                               populations=[S])
        new_result = BayesRGT(M=3,
                              T=1,
                              f=linear_bayesian_utility_model,
                              theta_star=theta_star,
                              thetas=[theta],
                              populations=[S_])
        self.assertGreaterEqual(new_result, prev_result)
