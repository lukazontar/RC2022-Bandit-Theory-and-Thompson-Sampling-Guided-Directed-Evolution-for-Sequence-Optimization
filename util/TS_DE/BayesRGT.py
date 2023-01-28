from typing import List, Callable
import numpy as np


def BayesRGT(M: int,
             T: int,
             f: Callable[[np.ndarray], float],
             theta_star: np.ndarray,
             thetas: List[np.ndarray],
             populations: List[List[np.ndarray]]) -> float:
    """
    Calculates the Bayesian regret for a given number of sequences evaluated per round M,
    function to be maximized f, thetas and populations.

    Args:
        M: Population size.
        T: Number of rounds.
        f: Protein utility function that we are trying to maximize. Function includes a parameter theta that we are optimizing.
        theta_star: optimal theta - parametrization of the linear Bayesian utility model for which we aim to optimize the protein design
        thetas: Linear utility function parametrizations for each step that was executed.
        populations: List of populations of sequences (protein motifs).

    Returns:
        Accumulated population average Bayesian regret for a given TS-DE execution.
    """

    accumulated_regret = 0
    # Go over all time steps - to accumulate the regret
    for t in range(T):
        if t == 0:
            continue
        S = populations[t]
        theta = thetas[t - 1]
        # Evaluate optimal utility - the maximum
        optimal_utility = max([f(x, theta) for x in S])
        # Go through all elements in a population
        for i in range(M):
            x_i = S[i]
            evaluated_utility = f(x_i, theta_star)
            accumulated_regret += (optimal_utility - evaluated_utility)

    # Return population average accumulated regret
    return accumulated_regret / M
