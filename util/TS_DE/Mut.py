import numpy as np


def Mut(x: np.ndarray,
        I: list[int],
        mu: float) -> np.ndarray:
    """
    Given a set of targeted sites, the targeted sites are (w.p. mu) mutated to follow unit uniform distribution.
    Defined in 3.3.

    Args:
        x: Parent sequence.
        I: Indices of features/sites where the single site fitness over the population is less than of a uniformly
        distributed sequence.
        mu: Mutation rate - (0, 1).

    Returns:
        Child sequence.

    Raises:
        ValueError if mu is not a probability value or if x is non-binary.
    """
    if mu > 1 or mu < 0:
        raise ValueError("mu is a probability value and should be in the interval between 0 and 1.")

    # We are testing the model on binary entries only!
    unique_vals = set(x)
    try:
        unique_vals.remove(0)
        unique_vals.remove(1)
        if len(unique_vals) > 0:
            raise ValueError("x has to be binary.")
    except KeyError:
        pass

    # Newly generated child sequence placeholder.
    z = x.copy()

    for i in I:
        # Mutate i-th site with probability mu.
        if np.random.uniform() < mu:

            z[i] = np.random.choice(a=[0, 1],
                                    size=1)[0]
        else:
            # This part can be omitted since z copies x in initialization. It is kept for transparency.
            z[i] = x[i]

    return z
