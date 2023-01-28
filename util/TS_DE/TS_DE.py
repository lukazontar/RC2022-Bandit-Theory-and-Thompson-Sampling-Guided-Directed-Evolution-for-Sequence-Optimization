from typing import Callable, List, Tuple
import numpy as np
from numpy.linalg import inv

from util.TS_DE.Crossover_Selection import Crossover_Selection
from util.TS_DE.Directed_Mutation import Directed_Mutation


def TS_DE(d: int,
          T: int,
          M: int,
          theta_star: np.ndarray,
          S_0: List[np.ndarray],
          mu: float,
          sigma: float,
          lambda_: float,
          f: Callable[[np.ndarray], float],
          verbose=False) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
    """
    Thompson Sampling-guided Directed Evolution algorithm with bandit method.
    TS-DE is an iterative process, where at each time step we:
    1. Update the posterior.
    2. Perform Thompson Sampling to generate theta_t.
    3. Perform directed mutation.
    4. Perform Crossover selection.
    5. Augment dataset for the next iteration with measurements of the new population.

    Args:
        theta_star: optimal theta - parametrization of the linear Bayesian utility model for which we aim to optimize the protein design
        verbose: Indicator whether log prints should be shown in the output (false by default).
        f: Protein utility function that we are trying to maximize. Function includes a parameter theta that we are optimizing.
        lambda_: Scalar that controls trade-off between exploitation and exploration in the optimization process.
        d: Number of protein motifs, eq. Sequence length.
        T: Number of rounds.
        M: Population size.
        S_0: Initial population consisting of M candidate sequences.
        mu: Mutation rate - (0, 1).
        sigma: Standard deviation used in Assumption 3.5.

    Returns:
        All generated populations and corresponding thetas for each step t. These can then be used to calculate
        Bayesian regret.
    """

    # Initialization
    # Variable D - representing the whole dataset should also be introduced here,
    # but since it is not used anywhere, we skipped that part.
    Phi = np.zeros((M, d))
    U = np.zeros((M, 1))

    # Initialize population
    S = S_0
    populations = [S_0]
    thetas = []
    # Iterate T times
    for i in range(T):
        # --------------------------------------------------------------------------------------------
        # 1. Update the posterior.
        # Calculating V
        Phi_dot_prod = np.transpose(Phi) @ Phi
        lambda_eye = lambda_ * np.eye(d)
        V = (1 / (sigma ** 2)) * Phi_dot_prod + lambda_eye
        # Calculating theta_hat
        inv_V_Phi_T = inv(V) @ Phi.T
        inv_V_Phi_T_U = inv_V_Phi_T @ U
        theta_hat = (1 / (sigma ** 2) * inv_V_Phi_T_U).reshape(-1)
        # --------------------------------------------------------------------------------------------
        # 2. Perform Thompson Sampling to generate theta_t.
        theta_tilde = np.random.multivariate_normal(mean=theta_hat,
                                                    cov=inv(V))

        # --------------------------------------------------------------------------------------------
        # 3. Perform directed mutation.
        num_ones_before_mut = np.array(S).sum()
        S = Directed_Mutation(d=d,
                              S=S,
                              theta=theta_tilde,
                              mu=mu)
        # --------------------------------------------------------------------------------------------
        # 4. Perform Crossover selection.
        num_ones_before_crossover_selection = np.array(S).sum()
        S = Crossover_Selection(f=f,
                                S=S,
                                theta=theta_tilde)
        num_ones_after_crossover_selection = np.array(S).sum()
        if verbose:
            print(
                f"delta 1s (mut) = {num_ones_before_crossover_selection - num_ones_before_mut}, delta 1s (cs) = {num_ones_after_crossover_selection - num_ones_before_crossover_selection}, 1s % = {num_ones_after_crossover_selection / (M * d)}")
        # --------------------------------------------------------------------------------------------
        # 5. Augment dataset for the next iteration with measurements of the new population.
        for i in range(len(S)):
            x_i = S[i]
            evaluated_utility = f(x=x_i, theta=theta_star)
            u_i = np.random.normal(loc=evaluated_utility, scale=sigma)
            Phi = np.append(arr=Phi,
                            values=x_i.reshape((1, -1)),
                            axis=0)

            U = np.append(arr=U,
                          values=np.array(u_i).reshape((-1, 1)),
                          axis=0)
        # Fill returning values that will be used to calculate Bayesian regret and fitness curves.
        populations.append(S)
        thetas.append(theta_tilde)
    return populations, thetas
