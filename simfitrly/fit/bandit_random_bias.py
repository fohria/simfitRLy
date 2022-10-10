import numpy as np
from scipy.optimize import minimize
from .model_store.random_bias.random_bias_bandit import likelihood


def fit_random_bias_single_subject_mle(actions, rewards, guesses_per_loop=10):
    """
    fit random_bias model to data from 2armbandit task, using scipy.minimize and provided actions, rewards

    Parameters
    ----------
    actions : np.ndarray
        list of all the actions taken by the subject
    rewards : np.ndarray
        list of all rewards received during the task
    guesses_per_loop : int, optional
        how many times to repeat minimize to find best result. The default is 10.

    Returns
    -------
    best_result : scipy.optimize.OptimizeResult
        fitresult dictionary object where:
            - parameter values are in the x key
            - log likelihood is in the fun key

    """

    best_loglike = 999999  # best log-likelihood will be much lower
    best_result = None
    rng = np.random.RandomState()

    for _ in range(guesses_per_loop):
        starting_guess = (rng.uniform(0, 1))  # bias
        fitresult = minimize(
            likelihood,
            starting_guess,
            args=(actions, rewards),
            bounds=[(0.00001, 1)],  # bias
        )
        if fitresult.fun < best_loglike and fitresult.success:
            best_result = fitresult
            best_loglike = fitresult.fun

    return best_result
