from typing import List
import numpy as np

from util.TS_DE.Mut import Mut


def Directed_Mutation(d: int,
                      S: List[np.ndarray],
                      theta: np.ndarray,
                      mu: float) -> List[np.ndarray]:
    """
    Finds sites that will be targeted for mutation and mutates them. Targeted sites are the ones where single site
    fitness over the population is less than of a uniformly distributed sequence. Defined on motif level.

    Args:
        d: Number of protein motifs, eq. Sequence length.
        S: A population of sequences.
        theta: Linear utility function parametrization.
        mu: Mutation rate - (0, 1).

    Returns:
        Child population S_.
    """
    I = set()
    S_ = list()
    # Size of the population
    M = len(S)

    # Find sites that will be targeted with mutation
    for i in range(d):
        col_x_i = np.array([x[i] for x in S])
        x_bar_i = np.array(S).mean()

        if (1 / M) * theta[i] * col_x_i.sum() <= theta[i] * x_bar_i:
            I.add(i)
    # On targeted sites, perform mutation
    for x in S:
        S_.append(Mut(x=x,
                      I=I,
                      mu=mu))

    return S_
