from typing import Tuple, List, Callable

import numpy as np

from util.DE.Random_Crossover_Selection import Random_Crossover_Selection
from util.DE.Random_Mutation import Random_Mutation


def DE(d: int,
       T: int,
       S_0: List[np.ndarray],
       theta_star: np.ndarray,
       mu: Callable[[int], float],
       sigma: float,
       f: Callable[[np.ndarray], float]

       ) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
    """
    Basic Directed Evolution algorithm.
    DE is an iterative process, where at each time step we:
    1. Perform random mutation.
    2. Perform random crossover recombination and selection.

    Args:
        sigma: Standard deviation.
        theta_star: optimal theta - parametrization of the linear Bayesian utility model for which we aim to optimize the protein design
        f: Protein utility function that we are trying to maximize. Function includes a parameter theta that we are optimizing.
        d: Number of protein motifs, eq. Sequence length.
        T: Number of rounds.
        S_0: Initial population consisting of M candidate sequences.
        mu: Mutation rate - (0, 1) - a function of T.

    Returns:
        All generated populations and corresponding thetas for each step t.
    """

    populations = [S_0]

    S = S_0.copy()
    # Iterate T times
    for t in range(T):
        # --------------------------------------------------------------------------------------------
        # 1. Perform random mutation.
        S = Random_Mutation(S=S,
                            mu=mu(t))
        # --------------------------------------------------------------------------------------------
        # 2. Perform Crossover selection.
        S = Random_Crossover_Selection(f=f,
                                       sigma=sigma,
                                       theta_star=theta_star,
                                       S=S)
        populations.append(S)

    return populations
