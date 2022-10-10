from cmdstanpy import CmdStanModel


def fit_single_subject_bandit(model_file, actions, rewards, subject_id, add_one=True):
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

    Returns
    -------
    fit : cmdstanpy.stanfit.CmdStanVB
        CmdStanVB object. use fit.variational_params_dict to see results
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

    # variational can fail due to init values so we retry until it works
    # adjusting init values doesn't really help but retrying does
    success = False
    retries = 0
    initvals = 2  # default as per https://cmdstanpy.readthedocs.io/en/v1.0.1/api.html
    while not success:
        if retries > 10:
            initvals = 5
        try:
            fit = model.variational(
                data=standata,
                output_dir=output_dir,
                inits=initvals,
                # grad_samples=250,
                # require_converged=False
            )
            success = True
        except:
            print("ELBO error, trying again...")
            retries += 1

    return fit, subject_id, retries
