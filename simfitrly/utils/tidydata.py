import numpy as np
import pandas as pd
import arviz as az


def tidy_dataframe(actions, rewards, stimuli, agent_params, task_params):
    """
    returns a tidy formatted pandas dataframe combining separate input parameters from a simulation.

    WARNING: remember this function works for single agent, task combinations

    """

    # haha i like this sneakyness
    if task_params.name == "reversal_bandit":
        return tidy_dataframe_reversal(
            actions, rewards, stimuli, agent_params, task_params
        )
    if task_params.name == "worthy_bandit":
        return tidy_dataframe_worthy(
            actions, rewards, stimuli, agent_params, task_params
        )

    if len(stimuli) != len(actions):
        raise ValueError("stimuli and action sequence should be same length")

    if len(rewards) != len(actions):
        raise ValueError("rewards and action sequence should be same length")

    subject_count = int(len(stimuli) / task_params.trial_count)
    data_length = len(stimuli)

    column_names = ["subject_id", "trial", "stimulus", "action", "reward"]
    parameter_values = []

    for parameter, value in task_params.get_dict().items():
        column_names.append("task_" + parameter)
        parameter_values.append(value)

    for parameter, value in agent_params.get_dict().items():
        column_names.append("agent_" + parameter)
        parameter_values.append(value)

    # assume all subjects use the same agent&task parameters
    params = np.tile(parameter_values, data_length)
    params = params.reshape([data_length, len(parameter_values)])

    # create individual ids and trial numbers for all subjects in data
    subject_ids = np.repeat(np.arange(subject_count), task_params.trial_count)
    trials = np.tile(np.arange(task_params.trial_count), subject_count)

    # reshape 1 dimensional vectors so we can stack everything later
    subject_ids = subject_ids.reshape(data_length, 1)
    trials = trials.reshape(data_length, 1)
    stimuli = stimuli.reshape(data_length, 1)
    actions = actions.reshape(data_length, 1)
    rewards = rewards.reshape(data_length, 1)

    # hstack will convert entire datarows into object type
    datarows = np.hstack((subject_ids, trials, stimuli, actions, rewards, params))

    df = pd.DataFrame(columns=column_names, data=datarows)
    # convert all numericals to numericals again, ignore the strings
    df = df.apply(pd.to_numeric, errors="ignore")

    # convert task and agent parameter columns to categories to save memory
    for column in df.columns:
        if "agent_" in column or "task_" in column:
            df[column] = df[column].astype("category")
    df["subject_id"] = df["subject_id"].astype("category")

    return df


def tidy_dataframe_reversal(actions, rewards, stimuli, agent_params, task_params):
    """
    annoyingly need special case of tidy dataframe to handle reversal switch points

    returns a tidy formatted pandas dataframe combining separate input parameters from a simulation.

    WARNING: remember this function works for single agent, task combinations

    """

    if len(stimuli) != len(actions):
        raise ValueError("stimuli and action sequence should be same length")

    if len(rewards) != len(actions):
        raise ValueError("rewards and action sequence should be same length")

    subject_count = int(len(stimuli) / task_params.trial_count)
    data_length = len(stimuli)

    column_names = ["subject_id", "trial", "stimulus", "action", "reward"]
    parameter_values = []

    for parameter, value in task_params.get_dict().items():
        column_names.append("task_" + parameter)
        parameter_values.append(value)

    for parameter, value in agent_params.get_dict().items():
        column_names.append("agent_" + parameter)
        parameter_values.append(value)

    # special case for reversal bandit here (also see below)
    parameter_values.pop(parameter_values.index(task_params.switch_points))
    column_names.pop(column_names.index("task_switch_points"))
    column_names.append("task_switch_points")  # need to reinsert at the end..

    # assume all subjects use the same agent&task parameters
    params = np.tile(parameter_values, data_length)
    params = params.reshape([data_length, len(parameter_values)])

    # create individual ids and trial numbers for all subjects in data
    subject_ids = np.repeat(np.arange(subject_count), task_params.trial_count)
    trials = np.tile(np.arange(task_params.trial_count), subject_count)

    # reshape 1 dimensional vectors so we can stack everything later
    subject_ids = subject_ids.reshape(data_length, 1)
    trials = trials.reshape(data_length, 1)
    stimuli = stimuli.reshape(data_length, 1)
    actions = actions.reshape(data_length, 1)
    rewards = rewards.reshape(data_length, 1)
    # special case for reversal bandit continued
    task_switch_points = [
        1 if trial in task_params.switch_points else 0
        for trial in range(task_params.trial_count)
    ]
    task_switch_points = np.tile(task_switch_points, subject_count)
    task_switch_points = np.array(task_switch_points).reshape(data_length, 1)

    # hstack will convert entire datarows into object type
    datarows = np.hstack(
        (subject_ids, trials, stimuli, actions, rewards, params, task_switch_points)
    )

    df = pd.DataFrame(columns=column_names, data=datarows)
    # another special case and ugly hacky solution
    df.task_arm1 = np.tile(task_params.arm1_list, subject_count)
    df.task_arm2 = np.tile(task_params.arm2_list, subject_count)
    # convert all numericals to numericals again, ignore the strings
    df = df.apply(pd.to_numeric, errors="ignore")

    # convert task and agent parameter columns to categories to save memory
    for column in df.columns:
        if "agent_" in column or "task_" in column:
            df[column] = df[column].astype("category")
    df["subject_id"] = df["subject_id"].astype("category")

    return df


def tidy_dataframe_worthy(actions, rewards, stimuli, agent_params, task_params):
    """
    returns a tidy formatted pandas dataframe combining separate input parameters from a simulation.

    WARNING: remember this function works for single agent, task combinations

    """

    if len(stimuli) != len(actions):
        raise ValueError("stimuli and action sequence should be same length")

    if len(rewards) != len(actions):
        raise ValueError("rewards and action sequence should be same length")

    subject_count = int(len(stimuli) / task_params.trial_count)
    data_length = len(stimuli)

    column_names = ["subject_id", "trial", "stimulus", "action", "reward"]
    parameter_values = []

    for parameter, value in task_params.get_dict().items():
        column_names.append("task_" + parameter)
        if parameter == "deck1" or parameter == "deck2":
            parameter_values.append(-1)  # placeholder
        else:
            parameter_values.append(value)

    for parameter, value in agent_params.get_dict().items():
        column_names.append("agent_" + parameter)
        parameter_values.append(value)

    # assume all subjects use the same agent&task parameters
    params = np.tile(parameter_values, data_length)

    # lengthen to entire dataset
    params = params.reshape([data_length, len(parameter_values)])

    # add back deck arrays instead of placeholders
    params[:, column_names.index("task_deck1") - 5] = np.tile(
        task_params.deck1, subject_count
    )
    params[:, column_names.index("task_deck2") - 5] = np.tile(
        task_params.deck2, subject_count
    )

    # create individual ids and trial numbers for all subjects in data
    subject_ids = np.repeat(np.arange(subject_count), task_params.trial_count)
    trials = np.tile(np.arange(task_params.trial_count), subject_count)

    # reshape 1 dimensional vectors so we can stack everything later
    subject_ids = subject_ids.reshape(data_length, 1)
    trials = trials.reshape(data_length, 1)
    stimuli = stimuli.reshape(data_length, 1)
    actions = actions.reshape(data_length, 1)
    rewards = rewards.reshape(data_length, 1)

    # hstack will convert entire datarows into object type
    datarows = np.hstack((subject_ids, trials, stimuli, actions, rewards, params))

    df = pd.DataFrame(columns=column_names, data=datarows)
    # convert all numericals to numericals again, ignore the strings
    df = df.apply(pd.to_numeric, errors="ignore")

    # convert task and agent parameter columns to categories to save memory
    for column in df.columns:
        if "agent_" in column or "task_" in column:
            df[column] = df[column].astype("category")
    df["subject_id"] = df["subject_id"].astype("category")

    return df


def tidy_mle(model, fitdata, simdata):
    """
    create tidy dataframe for fitresults
    NOTE: currently fitdata has to be numpy array
    TODO: if human data, simdata is a bad variable name

    Parameters
    ----------
    model : Model object
        the model used to create `fitdata`.
    fitdata : numpy array
        the data from fitting process, with subject_id, loglike and parameter values as columns, each subject is one row
    simdata : pandas dataframe
        dataset that was used for fitting, will be in shape of tidy_dataframe

    Returns
    -------
    df : pandas dataframe
    will have one row for each subject, with columns combined from simdata and fitdata, but excluding all the actions, rewards and stimuli

    """

    modelparams = model.parameters
    param_count = len(modelparams)
    trial_count = len(simdata.trial.unique())
    print(f"using trial_count {trial_count} for BIC")
    print(f"using param_count {param_count} for BIC")

    # copy simdata and get one row per subject
    df = simdata.query("trial == 1")  # query returns copy
    try:  # bandit task don't always have stimulus, especially human datasets
        df = df.drop(columns=["trial", "stimulus", "action", "reward"])
    except:
        df = df.drop(columns=["trial", "action", "reward"])
    df = df.reset_index(drop=True)

    columns = ["subject_id", "loglike"]
    for parameter in modelparams:
        columns.append(f"fit{parameter}")
    fitdata = pd.DataFrame(columns=columns, data=fitdata)
    fitdata.subject_id = fitdata.subject_id.astype(int)

    # merge and add info columns
    fitdata = df.merge(fitdata)
    fitdata["method"] = "mle"
    fitdata["model_name"] = model.name
    fitdata["bic"] = fitdata.loglike.apply(
        lambda x: param_count * np.log(trial_count) - 2 * x
    )

    return fitdata


# TODO: tidy_vb and tidy_mle are the same basically


def tidy_vb(model, fitdata, simdata):
    """
    create tidy dataframe for fitresults
    NOTE: currently fitdata has to be numpy array

    Parameters
    ----------
    model : Model object
        the model used to create `fitdata`.
    fitdata : numpy array
        the data from fitting process, with subject_id, loglike and parameter values as columns, each subject is one row
    simdata : pandas dataframe
        dataset that was used for fitting, will be in shape of tidy_dataframe

    Returns
    -------
    df : pandas dataframe
    will have one row for each subject, with columns combined from simdata and fitdata, but excluding all the actions, rewards and stimuli

    """

    modelparams = model.parameters
    param_count = len(modelparams)

    # copy simdata and get one row per subject
    df = simdata.query("trial == 1")  # query returns copy
    df = df.drop(columns=["trial", "stimulus", "action", "reward"])
    df = df.reset_index(drop=True)

    columns = ["subject_id", "loglike"]
    for parameter in modelparams:
        columns.append(f"fit{parameter}")
    fitdata = pd.DataFrame(columns=columns, data=fitdata)
    fitdata.subject_id = fitdata.subject_id.astype(int)

    # merge and add info columns
    fitdata = df.merge(fitdata)
    fitdata["method"] = "vbstan"
    fitdata["model_name"] = model.name
    fitdata["bic"] = fitdata.loglike.apply(
        lambda x: param_count * np.log(fitdata.task_trial_count.iloc[0]) - 2 * x
    )

    return fitdata


def tidy_mcmc(modelparams, fitdata, simdata):

    model_params = modelparams["param_names"]
    rows = []
    for subject, result in enumerate(fitdata):

        # get all "real" param values for this simulated subject
        subject_data = simdata[simdata.subject_id == subject].iloc[0]
        sim_param_values = [
            subject_data[f"agent_{parameter}"]
            for parameter in model_params
            if f"agent_{parameter}" in subject_data.index
        ]

        # if fitmodel != sim model or subjects are human we dont have (all) real param values
        if len(sim_param_values) == 0:
            sim_param_values = [None for _ in model_params]
        elif len(sim_param_values) < len(model_params):
            nones = [None for _ in range(len(model_params) - len(sim_param_values))]
            sim_param_values.extend(nones)

        # get parameter estimates from fit results
        subject_fit_summary = az.summary(result, kind="stats")
        param_fits = []
        for parameter in model_params:
            param_data = subject_fit_summary.loc[parameter].copy()
            param_data = param_data.rename(
                {
                    "mean": f"{parameter}",
                    "sd": f"{parameter}_sd",
                    "hdi_3%": f"{parameter}_hdi3",
                    "hdi_97%": f"{parameter}_hdi97",
                }
            )
            param_fits.append(param_data)
        # create dataframe row for this subject's parameter estimates
        subject_fit_row = pd.concat(param_fits).to_frame().T

        # add subject id to row
        subject_fit_row["subject_id"] = subject

        # add real parameter values (or None if it doesn't exist) for all model parameters
        for param, real_value in zip(model_params, sim_param_values):
            subject_fit_row[f"{param}_real"] = real_value

        # add subject to rows
        rows.append(subject_fit_row)

    fit_data = pd.concat(rows).reset_index(drop=True)
    return fit_data
