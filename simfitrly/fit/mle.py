import numpy as np
from scipy.optimize import minimize


def fit_single_subject_bandit(
    model_likelihood,
    starting_guess_bounds,
    fit_bounds,
    actions,
    rewards,
    guesses_per_loop=10,
):
    """
    fit `model_likelihood` to data from 2armbandit task, using scipy.minimize and provided actions, rewards

    Parameters
    ----------
    model_likelihood : function
        the likelihood function for the model we want to fit
    starting_guess : list of tuples
        starting guess limits for parameter values
    bounds : list of tuples
        bounds for the parameter values
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

    while guesses_per_loop > 0 or best_result is None:

        start_guess = [
            rng.uniform(lower, upper) for lower, upper in starting_guess_bounds
        ]

        fitresult = minimize(
            model_likelihood,
            start_guess,
            args=(actions, rewards),
            bounds=fit_bounds,
        )

        if fitresult.fun < best_loglike and fitresult.success:
            best_result = fitresult
            best_loglike = fitresult.fun

        guesses_per_loop -= 1

    return best_result


def fit_single_subject_shapetask(
    model_likelihood,
    starting_guess_bounds,
    fit_bounds,
    actions,
    rewards,
    stimuli,
    guesses_per_loop=10,
):
    """
    fit `model_likelihood` to data from shapetask, using scipy.minimize and provided actions, rewards, stimuli

    Parameters
    ----------
    model_likelihood : function
        the likelihood function for the model we want to fit
    starting_guess : list of tuples
        starting guess limits for parameter values
    bounds : list of tuples
        bounds for the parameter values
    actions : np.ndarray
        list of all the actions taken by the subject
    rewards : np.ndarray
        list of all rewards received during the task
    stimuli : np.ndarray
        list of all stimuli used in the task
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

    while guesses_per_loop > 0 or best_result is None:

        start_guess = [
            rng.uniform(lower, upper) for lower, upper in starting_guess_bounds
        ]

        fitresult = minimize(
            model_likelihood,
            start_guess,
            args=(actions, rewards, stimuli),
            bounds=fit_bounds,
        )

        if fitresult.fun < best_loglike and fitresult.success:
            best_result = fitresult
            best_loglike = fitresult.fun

        guesses_per_loop -= 1

    return best_result


def fit_single_subject_shapetask_HRL(
    model_likelihood,
    starting_guess_bounds,
    fit_bounds,
    actions,
    rewards,
    stimuli,
    taskset_count=1000,
):

    rng = np.random.RandomState()

    best_results = []

    for _ in range(taskset_count):

        tasksets = rng.randint(low=0, high=2, size=99)  # high is non inclusive

        guesses_per_loop = 10
        best_loglike = 999999  # best log-likelihood will be much lower
        best_result = None

        while guesses_per_loop > 0 or best_result is None:

            start_guess = [
                rng.uniform(lower, upper) for lower, upper in starting_guess_bounds
            ]

            fitresult = minimize(
                model_likelihood,
                start_guess,
                args=(actions, rewards, stimuli, tasksets),
                bounds=fit_bounds,
            )

            if fitresult.fun < best_loglike and fitresult.success:
                best_result = fitresult
                best_loglike = fitresult.fun

            guesses_per_loop -= 1

        best_results.append(best_result)

    results = np.array(
        [[fit.fun, fit.x[0], fit.x[1], fit.x[2], fit.x[3]] for fit in best_results]
    )
    res_mean = np.mean(results, axis=0)
    # res_median = np.median(results, axis=0)
    # return res_mean, res_median
    return hrl_fit(res_mean[0], res_mean[1:])


from dataclasses import dataclass, field


@dataclass
class hrl_fit:

    fun: float
    x: list = field(default_factory=list)
