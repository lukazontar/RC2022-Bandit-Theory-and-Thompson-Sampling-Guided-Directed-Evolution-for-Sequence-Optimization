import unittest
import numpy as np

from util.common import linear_bayesian_utility_model
from util.TS_DE.BayesRGT import BayesRGT


def example_BayesRGT():
    np.random.seed(seed=0)
    M = 5
    T = 5
    d = 4

    theta_star = np.random.multivariate_normal(np.zeros(d), np.eye(d))

    thetas = [np.array([0.1, 0.2, 0.3, 0.4]),
              np.array([0.2, 0.3, 0.4, 0.5]),
              np.array([0.3, 0.4, 0.5, 0.6]),
              np.array([0.4, 0.5, 0.6, 0.7]),
              np.array([0.5, 0.6, 0.7, 0.8])]
    populations = [[np.array([0, 0, 0, 0]),
                    np.array([0, 0, 0, 0]),
                    np.array([0, 0, 0, 0]),
                    np.array([0, 0, 0, 0]),
                    np.array([0, 0, 0, 0])],
                   [np.array([1, 0, 1, 1]),
                    np.array([0, 1, 0, 1]),
                    np.array([1, 0, 1, 0]),
                    np.array([0, 1, 0, 0]),
                    np.array([1, 0, 0, 0])],
                   [np.array([1, 1, 1, 1]),
                    np.array([1, 1, 0, 1]),
                    np.array([1, 0, 1, 1]),
                    np.array([0, 1, 1, 1]),
                    np.array([1, 1, 1, 0])],
                   [np.array([1, 1, 1, 1]),
                    np.array([1, 1, 1, 1]),
                    np.array([1, 1, 1, 1]),
                    np.array([1, 1, 1, 1]),
                    np.array([1, 1, 1, 1])],
                   [np.array([1, 1, 1, 1]),
                    np.array([1, 1, 1, 1]),
                    np.array([1, 1, 1, 1]),
                    np.array([1, 1, 1, 1]),
                    np.array([1, 1, 1, 1])],
                   [np.array([1, 1, 1, 1]),
                    np.array([1, 1, 1, 1]),
                    np.array([1, 1, 1, 1]),
                    np.array([1, 1, 1, 1]),
                    np.array([1, 1, 1, 1])]]
    sigma = 0.1
    result = BayesRGT(M=M,
                      T=T,
                      theta_star=theta_star,
                      f=linear_bayesian_utility_model,
                      thetas=thetas,
                      populations=populations)
    return result


class TestBayesRGT(unittest.TestCase):
    def test_output_value(self):
        expected_result = -11.38
        result = example_BayesRGT()

        self.assertAlmostEqual(result, expected_result, delta=1e-2)

    def test_output_type(self):
        result = example_BayesRGT()

        self.assertIsInstance(result, float)
