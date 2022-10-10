from cmdstanpy import CmdStanModel


def fit_single_subject_bandit(
    model_file, actions, rewards, subject_id, add_one=True, parallel_chains=1):
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
    
    print(f"starting mcmc fit procedure for subject {subject_id}")
    output_dir = f"data/fits/temp/{subject_id}"

    fit = model.sample(
        data=standata,
        chains=4,
        parallel_chains=1,
        threads_per_chain=1,
        iter_warmup=1000,
        iter_sampling=5000,
        show_progress=False,
        output_dir=output_dir
    )

    return fit, subject_id
