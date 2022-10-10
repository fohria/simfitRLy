import numpy as np
import multiprocessing as mp
from itertools import product
import pandas as pd

# from .agents import QL
from simfitrly.utils.tidydata import tidy_dataframe

# TODO fix up this file so it can handle any task/agent combo


def simulate_many(agent, task, subject_repetitions=100):
    """
    simulate `agent` playing `task` and do it `subject_repetitions` many times
    """

    results = []

    for subject in range(subject_repetitions):
        result = agent.play(task)
        results.append(result)

    all_actions = np.ravel([result["actions"] for result in results])
    all_rewards = np.ravel([result["rewards"] for result in results])
    all_stimuli = np.ravel([result["stimuli"] for result in results])

    df = tidy_dataframe(all_actions, all_rewards, all_stimuli, agent, task)
    df = task.score_experiment(df)

    return df


def simulate_simset(agent, task, subject_count, simset_id=-1):
    """

    generate data using `agent` playing `task`.
    one simset is a set of simulations all using the same agent and task
    TODO: can currently only play bandit task

    Parameters
    ----------
    agent : AgentParameters
        parameters for the agent.
    task : TaskParameters
        parameters for the task.
    subject_count : int
        how many subjects to simulate in this simset
    simset_id : int, optional
        what simset #id to use in the dataframe returned, default is -1

    Returns
    -------
    simset : pandas dataframe
        simulated data for all `subject_count` subjects.

    """

    all_stimuli = []
    all_actions = []
    all_rewards = []

    for subject in range(subject_count):

        results = agent.play(task)

        all_stimuli.append(results["stimuli"])
        all_actions.append(results["actions"])
        all_rewards.append(results["rewards"])

    all_stimuli = np.concatenate(all_stimuli)
    all_actions = np.concatenate(all_actions)
    all_rewards = np.concatenate(all_rewards)

    simset = tidy_dataframe(all_actions, all_rewards, all_stimuli, agent, task)

    simset = task.score_experiment(simset)
    simset["simset"] = simset_id

    return simset


def simulate_many_simsets(task, agent_count=10, subject_count=100, param_ranges=None):
    """

    generate `agent_count` simsets, each using `subject_count` participants
    one simset is a set of simulations all using the same agent and task

    TODO: currently this function uses specific task
    TODO: currently this function uses specific QL2
    TODO: could have task as input, and what kind of agent generation wanted
          where we have set if random params or param ranges we combine

    Parameters
    ----------
    task : Bandit
        object containing the task settings
    agent_count : int, optional
        how many different AgentParameters to use. The default is 10.
    subject_count : int, optional
        how many subjects for each AgentParameter. The default is 100.
    param_ranges : tuple of 3 numpy arrays, optional
        if provided, simulate using all possible permutations of these parameter ranges

    Returns
    -------
    simsets : pandas dataframe
        all the simsets in one big dataframe.

    """

    if param_ranges is None:
        agents = [
            QL(
                alpha=np.random.uniform(0, 1),
                beta=np.random.uniform(1, 20),
                gamma=0,
            )
            for _ in range(agent_count)
        ]
    else:
        # alphas = np.arange(0.01, 1.01, 0.02)
        # betas = np.array([1, 2, 5, 10, 20])
        # gammas = np.array([0])
        # TODO: if we consolidate functions, we can loop through param_ranges here
        alphas = param_ranges[0]
        betas = param_ranges[1]
        gammas = param_ranges[2]

        agents = [QL(*params) for params in product(alphas, betas, gammas)]

    simlist = (
        (agent, task, subject_count, simset_id)
        for simset_id, agent in enumerate(agents)
    )

    with mp.Pool() as pool:
        simsets = pool.starmap(simulate_simset, simlist)

    return pd.concat(simsets)


#
#
#
#
#

# so this function is quite silly, but can be useful as abstraction
def simulate_one(agent, task):
    actions, rewards, stimuli = agent.play_bandit(task)
    return actions, rewards, stimuli
