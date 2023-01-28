import numpy as np


def Rcb(x: np.ndarray,
        y: np.ndarray) -> np.ndarray:
    """
    Generates a child sequence z from a random pair of parent sequences x, y,
    where z_i is either x_i or y_i w.p. 1/2, respectively.
    Article: Definition 3.4
    Args:
        x: Parent sequence 1.
        y: Parent sequence 2.

    Returns:
        Child sequence.

    Raises:
        Value error: if x and y are not of same length or if any is not binary.
    """
    if len(x) != len(y):
        raise ValueError("x and y have to be of the same length.")

    # We are testing the model on binary entries only!
    unique_vals = set(x).union(set(y))
    try:
        unique_vals.remove(0)
        unique_vals.remove(1)
        if len(unique_vals) > 0:
            raise ValueError("x and y have to be binary.")
    except KeyError:
        pass

    # Newly generated child sequence placeholder
    z = list()

    for i in range(len(x)):
        # Uniformly sample z_i from x or y.
        theta = np.random.uniform()
        z.append(
            x[i] if theta < 0.5 else y[i]
        )
    return np.array(z)
