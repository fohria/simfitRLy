import numpy as np
import arviz as az

from scipy.optimize import minimize
from pandas import DataFrame
from cmdstanpy import CmdStanModel

from .model_store.ql2.ql2_bandit import likelihood


def fit_ql2_single_subject_mcmc(
    actions, rewards, subject_id, model_file, add_one=True, parallel_chains=1
):
    """
    fit QL2 model to individual subject data from 2armbandit task
    uses MCMC method for estimation based on provided actions, rewards

    Parameters
    ----------
    actions : np.ndarray
        list of all the actions taken by the subject
    rewards : np.ndarray
        list of all rewards received during the task
    subject_id : int
        for multiprocessing, cmdstan needs separate folders or collisions occur
    add_one : bool, optional
        stan uses 1-based indeces so we convert from 0,1 to 1,2
    parallel_chains : int, optional
        how many chains to run in parallel. The default is 1.

    Returns
    -------
    fit : cmdstanpy.stanfit.CmdStanMCMC
        CmdStanMCMC object with fit results. Use arviz to summarize
    subject_id : int
        to make results easy to track the subject_id is also returned

    """

    trial_count = len(actions)
    if add_one:
        actions = actions + 1

    standata = {
        "trial_count": trial_count,
        "action_seq": actions,
        "reward_seq": rewards,
    }

    # model_file = "fit/model_store/ql2/ql2_bandit_new.stan"
    model = CmdStanModel(stan_file=model_file)

    print(f"starting fit procedure for subject {subject_id}")
    output_dir = f"data/temp/{subject_id}"

    fit = model.sample(
        data=standata,
        chains=4,
        parallel_chains=1,
        threads_per_chain=1,
        iter_warmup=1000,
        iter_sampling=5000,
        show_progress=False,
        output_dir=output_dir,
    )

    return fit, subject_id


def fit_ql2_single_subject_mle(actions, rewards, guesses_per_loop=10):
    """
    fit QL2 model to data from 2armbandit task, using scipy.minimize and provided
    actions, rewards

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
        starting_guess = (rng.uniform(0, 1), rng.uniform(1, 20))  # alpha, beta
        fitresult = minimize(
            likelihood,
            starting_guess,
            args=(actions, rewards),
            bounds=[(0.001, 1), (1, 40)],  # alpha, beta
        )
        if fitresult.fun < best_loglike and fitresult.success:
            best_result = fitresult
            best_loglike = fitresult.fun

    return best_result
