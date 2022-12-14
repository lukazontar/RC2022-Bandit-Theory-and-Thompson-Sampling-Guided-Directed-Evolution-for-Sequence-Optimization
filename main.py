def Mut(x,
        I: list[int],
        mu: float):
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
    """
    raise NotImplementedError


def Rcb(x,
        y):
    """
    Generates a child sequence z from a random pair of parent sequences x, y, where z_i is either x_i or y_i w.p. 1/2,
    respectively.
    Article: Definition 3.4
    Args:
        x: Parent sequence 1.
        y: Parent sequence 2.

    Returns:
        Child sequence.
    """
    raise NotImplementedError


def Crossover_Selection(f,
                        S):
    """
    Generates a set of recombinations that perform better than the parents' average.
    Args:
        f: Utility function.
        S: A population of sequences.

    Returns:
        Child population S_.
    """
    raise NotImplementedError


def Directed_Mutation(f,
                      S,
                      mu: float):
    """
    Finds sites that will be targeted for mutation and mutates them. Targeted sites are the ones where single site
    fitness over the population is less than of a uniformly distributed sequence.
    Args:
        f: Utility function.
        S: A population of sequences.
        mu: Mutation rate - (0, 1).

    Returns:
        Child population S_.
    """
    raise NotImplementedError


def BayesRGT(T: int,
             M: int):
    """

    Args:
        T: Number of rounds.
        M: Population size

    Returns:

    """
    raise NotImplementedError


def TS_DE(T: int,
          S_0,
          M: int,
          mu: float,
          stdev):
    """
    Thompson Sampling-guided Directed Evolution algorithm with bandit method.
    TS-DE is an iterative process, where at each time step we:
    1. Update the posterior.
    2. Perform Thompson Sampling to generate theta_t.
    3. Perform directed mutation.
    4. Perform Crossover selection.
    5. Augment dataset for the next iteration with measurements of the new population.
    Args:
        T: Number of rounds.
        S_0: Initial population.
        M: Population size
        mu: Mutation rate - (0, 1).
        stdev: Standard deviation used in Assumption 3.5.

    Returns:
        Augmented dataset.
    """
    # Iterate T times
    for i in range(T):
        print(i)    # TODO: delete this placeholder.
        # 1. Update the posterior.
        # 2. Perform Thompson Sampling to generate theta_t.
        # 3. Perform directed mutation.
        # 4. Perform Crossover selection.
        # 5. Augment dataset for the next iteration with measurements of the new population.
    raise NotImplementedError