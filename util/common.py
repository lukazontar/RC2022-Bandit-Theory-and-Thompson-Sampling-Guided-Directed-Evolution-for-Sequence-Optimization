from typing import List
import numpy as np


def linear_bayesian_utility_model(x: np.ndarray,
                                  theta: np.ndarray) -> float:
    """
    Utility function used to model protein fitness. Instead of actual sequences, we use motifs since they tend
    to evolve as a whole.
    Args:
        x: Protein motif sequence encoded as an array of 0s and 1s, where 0s and 1s correspond to non-favorable and favorable positions, respectively.
        theta: Linear function parametrization. As per Assumption 3.2. it is sampled from a Gaussian prior.

    Returns:
        Protein utility, fitness.
    """
    return np.inner(theta, x)


def zero_population(d: int,
                    M: int) -> List[np.ndarray]:
    """
    Generates an initial population of all 0s with M candidate sequences, where each sequence has d motifs.
    Args:
        d: Number of protein motifs, eq. Sequence length.
        M: Population size.

    Returns:
        Initial population S_0.
    """
    S_0 = np.zeros((M, d)).tolist()
    S_0 = [np.array(x) for x in S_0]
    return S_0


def random_init_population(d: int,
                           M: int) -> List[np.ndarray]:
    """
    Generates an initial population of random 0s and 1s with M candidate sequences, where each sequence has d motifs.
    Args:
        d: Number of protein motifs, eq. Sequence length.
        M: Population size.

    Returns:
        Initial population S_0.
    """
    S_0 = [
        np.random.choice(a=[0, 1],
                         size=d)
        for _ in range(M)
    ]
    return S_0
