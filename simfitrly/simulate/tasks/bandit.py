"""
    bandit task.
    in this task, there are no stimuli, only 2 arms to pull.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .task import Task


@dataclass
class Bandit(Task):
    """
    2 armed bandit task

        arm1       : probability of reward on arm1
        arm2       : probability of reward on arm2
        trial_count: number of experimental trials
        name       : name of the task/experiment
    """

    arm1: float
    arm2: float
    trial_count: int
    name: str = "2arm_bandit"

    def get_stimuli(self):
        return generate_stimuli_sequence(self)

    def score_experiment(self, dataframe):
        return score_experiment(dataframe, self)


def generate_stimuli_sequence(taskparams):
    """
    bandit task doesn't have stimuli, so this returns None
    """

    stimuli = [None for _ in range(taskparams.trial_count)]
    return np.array(stimuli)


def score_experiment(df, taskparams):
    """
    calculate relevant scores for this experimental task.
    df should be a tidy dataframe.

    scoring:
        correct - means picking the arm with highest reward probability
        winstay - if the last action was rewarded, stay with that action
        loseshift - if the last action gave no reward, shift action to new one
    """

    # correct score
    arms = [taskparams.arm1, taskparams.arm2]
    best_arm = np.argmax(arms)
    df["correct"] = df.action == best_arm

    # winstay score
    stay = df.action.shift(1) == df.action  # previous action == this action
    win = df.reward.shift(1)  # was last action rewarded?
    winstay = win + stay  # if both are true we get sum of 2
    df["winstay"] = winstay == 2
    df.loc[df.trial == 0, "winstay"] = np.nan  # can't winstay on first trial
    df.winstay = pd.to_numeric(df.winstay)  # don't want series type to be object

    # loseshift score
    df["loseshift"] = winstay == 0
    df.loc[df.trial == 0, "loseshift"] = np.nan
    df.loseshift = pd.to_numeric(df.loseshift)

    return df
