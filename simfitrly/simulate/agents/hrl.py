from dataclasses import dataclass
import numpy as np
from numba import njit, int64

from .agent import Agent
from ..utils.choose import choose


@dataclass
class HRL(Agent):
    """
    HRL agent implementation based on Eckstein & Collins 2020

    context: position/shapetype/lastinbag
    """

    alpha_low: float
    alpha_high: float
    beta_low: float
    beta_high: float
    context: str = "position"
    name: str = "HRL"

    def play(self, task):
        if task.name == "shapetask":
            return self.play_shapetask(task)
        else:
            raise ValueError("that task not implemented for this agent!")

    def play_shapetask(self, taskparams):
        if self.context == "position":
            stimuli = taskparams.get_stimuli()
            contexts = np.tile([0, 1, 2], taskparams.trial_count // taskparams.bagsize)
        if self.context == "shapetype":
            contexts = taskparams.get_stimuli()
            stimuli = np.tile([0, 1, 2], taskparams.trial_count // taskparams.bagsize)
        if self.context == "lastinbag":
            stimuli = taskparams.get_stimuli()
            contexts = np.tile([0, 0, 1], taskparams.trial_count // taskparams.bagsize)
        actions, rewards, tasksets = play_shapetask(
            self.alpha_low,
            self.alpha_high,
            self.beta_low,
            self.beta_high,
            taskparams.trial_count,
            stimuli,
            contexts,
        )
        return {
            "actions": actions,
            "rewards": rewards,
            "stimuli": stimuli,
            "contexts": contexts,
            "tasksets": tasksets,
        }


@njit
def play_shapetask(
    alpha_low: float,
    alpha_high: float,
    beta_low: float,
    beta_high: float,
    trial_count: int,
    stimuli: np.array,
    contexts: np.array,
):

    # init qvalues for both hierarchical levels
    context_size = len(np.unique(contexts))
    if context_size == 3:
        q_high = np.ones((3, 3)) * 1 / 3  # 3 task sets: one for each position
        q_low = np.ones((3, 3, 3)) * 1 / 3  # 3 tasksets, 3 shapes, 3 actions
    elif context_size == 2:
        q_high = np.ones((2, 2)) * 0.5  # 2 task sets: one for each context
        q_low = np.ones((2, 3, 3)) * 1 / 3  # 2tasksets, 3 shapes, 3 actions

    # storage
    actions = np.zeros(trial_count, dtype=int64)
    rewards = np.zeros(trial_count, dtype=int64)
    tasksets = np.zeros(trial_count, dtype=int64)  # maybe call this high level action?

    for trial, (stimulus, context) in enumerate(zip(stimuli, contexts)):

        # select task set based on context and q_high values
        q_high_row = q_high[context]
        p_high = softmax(beta_high * q_high_row)
        # taskset = np.random.choice([0, 1, 2], p=p_high)
        taskset = choose([0, 1, 2], p_high)

        # select action based on q_low values for selected task set
        q_low_row = q_low[taskset, stimulus]
        p_low = softmax(beta_low * q_low_row)
        # action = np.random.choice([0, 1, 2], p=p_low)
        action = choose([0, 1, 2], p_low)

        # store selected taskset, action
        actions[trial] = action
        tasksets[trial] = taskset

        # cant get reward or next state for last trial
        if trial == trial_count - 1:
            # rewards[trial] = -1
            break

        # % get reward if action is correct (predicting next shape)
        if action == stimuli[trial + 1]:
            reward = 1
        else:
            reward = 0
        rewards[trial] = reward

        # % calculate reward prediction error for low values
        rpe_low = reward - q_low_row[action]
        q_low[taskset, stimulus, action] += alpha_low * rpe_low

        # % calculate reward prediction error for high values
        rpe_high = reward - q_high_row[taskset]
        q_high[context, taskset] += alpha_high * rpe_high

    return actions, rewards, tasksets


@njit
def softmax(array):
    exp_qvals = np.exp(array)
    probs = exp_qvals / np.sum(exp_qvals)
    return probs
