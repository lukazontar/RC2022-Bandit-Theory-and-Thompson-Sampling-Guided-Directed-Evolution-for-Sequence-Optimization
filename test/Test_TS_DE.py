import unittest
import numpy as np
from typing import List, Callable

from util.common import linear_bayesian_utility_model
from util.TS_DE.TS_DE import TS_DE


class TestTSDE(unittest.TestCase):
    def test_output_value(self):
        np.random.seed(0)

        d = 4
        T = 2
        M = 2
        S_0 = [
            np.array([1, 1, 1, 1]),
            np.array([0, 0, 0, 0])
        ]
        mu = 0.1
        sigma = 0.1
        lambda_ = 0.1
        f = linear_bayesian_utility_model
        theta_star = np.random.multivariate_normal(np.zeros(d), np.eye(d))

        populations, thetas = TS_DE(d=d, T=T, M=M, theta_star=theta_star, S_0=S_0, mu=mu, sigma=sigma, lambda_=lambda_,
                                    f=f)
        expected_result = [np.array([0, 0, 1, 0]), np.array([0, 0, 1, 0])]

        self.assertTrue(np.array_equal(np.array(expected_result), np.array(populations[-1])))


    def test_output_type(self):
        np.random.seed(0)

        d = 4
        T = 10
        M = 5
        S_0 = [np.array([0, 1, 0, 1]),
               np.array([1, 1, 0, 0]),
               np.array([1, 0, 0, 1]),
               np.array([0, 0, 1, 1]),
               np.array([1, 1, 1, 1])]
        mu = 0.5
        sigma = 0.2
        lambda_ = 0.5
        f = linear_bayesian_utility_model
        theta_star = np.random.multivariate_normal(np.zeros(d), np.eye(d))

        result = TS_DE(d=d, T=T, M=M, theta_star=theta_star, S_0=S_0, mu=mu, sigma=sigma, lambda_=lambda_, f=f)
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], list)
        self.assertIsInstance(result[1], list)
        self.assertIsInstance(result[0][0], list)
        self.assertIsInstance(result[0][0][0], np.ndarray)
