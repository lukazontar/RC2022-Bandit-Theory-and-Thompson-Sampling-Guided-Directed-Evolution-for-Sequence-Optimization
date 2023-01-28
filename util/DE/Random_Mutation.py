from typing import List

import numpy as np


def Random_Mutation(S: List[np.ndarray],
                    mu: float) -> List[np.ndarray]:
    """
    Finds sites that will be targeted for mutation and mutates them. Targeted sites are the ones where single site
    fitness over the population is less than of a uniformly distributed sequence. Defined on motif level.

    Args:
        S: A population of sequences.
        mu: Mutation rate - (0, 1).

    Returns:
        Child population S_.
    """
    S_ = list()

    # Perform mutation randomly
    for x in S:
        z = x.copy()
        for i in range(len(x)):
            # Mutate i-th site with probability mu.
            if np.random.uniform() < mu:
                z[i] = np.random.choice(a=[0, 1],
                                        size=1)[0]
            else:
                # This part can be omitted since z copies x in initialization. It is kept for transparency.
                z[i] = x[i]
        S_.append(z)
    return S_
