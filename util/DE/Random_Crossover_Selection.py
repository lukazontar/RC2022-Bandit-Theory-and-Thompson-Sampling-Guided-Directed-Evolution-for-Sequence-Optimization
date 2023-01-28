from typing import Callable

import numpy as np

from util.Common.Rcb import Rcb


def Random_Crossover_Selection(S: list[np.ndarray],
                               theta_star: np.ndarray,
                               sigma: float,
                               f: Callable[[np.ndarray], float]) -> np.ndarray:
    """
    Generates a set of recombinations that perform better than the parents' average. Defined on motif level.

    Args:
        sigma: Standard deviation.
        theta_star: optimal theta - parametrization of the linear Bayesian utility model for which we aim to optimize the protein design
        f: Utility function.
        S: A population of sequences.

    Returns:
        Child population S_.
    """

    S_ = list()
    # Until we generate enough sequences
    while len(S_) < len(S):
        # We sample random sequence x and y uniformly
        ix_x, ix_y = np.random.choice(a=range(len(S)),
                                      size=2,
                                      replace=True)
        x, y = S[ix_x], S[ix_y]

        # Perform recombination
        z = Rcb(x=x,
                y=y)

        # If utility of z is higher than average utility of its parents, include z in the new population.
        utility_z = np.random.normal(f(x=z, theta=theta_star), sigma)
        utility_x = np.random.normal(f(x=x, theta=theta_star), sigma)
        utility_y = np.random.normal(f(x=y, theta=theta_star), sigma)
        avg_utility_x_y = (utility_x + utility_y) / 2
        if utility_z >= avg_utility_x_y:
            S_.append(z)

    return S_
