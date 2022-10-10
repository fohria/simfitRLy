"""
    reversal bandit task.
    in this task, there are no stimuli, only 2 arms to pull.
    at switch points, the arm reward probabilities change
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .task import Task


@dataclass
class ReversalBandit(Task):
    """
    2 armed reversal bandit task. on creation, arm1 and arm2 will be combined with
    switch_points to create additional arm1, arm2 arrays for entire trial_count length.
    these arrrays are called arm1_list and arm2_list respectively and are useful when
    simulating the task.

        arm1         : probability of reward on arm1
        arm2         : probability of reward on arm2
        trial_count  : total number of trials
        switch_points: on these trials, arm1 and arm2 will switch
        name         : name of the task (default is reversal_bandit)
    """

    arm1: float
    arm2: float
    trial_count: int
    switch_points: list
    name: str = "reversal_bandit"

    def __post_init__(self):
        arm1_list, arm2_list = create_arm_arrays(self)

        self.arm1_list = arm1_list
        self.arm2_list = arm2_list

        # these are needed or tidy_dataframe_reversal will fail because it tries
        # to index and pop numpy values (sigh)
        self.arm1 = float(self.arm1)
        self.arm2 = float(self.arm2)
        self.trial_count = int(self.trial_count)
        self.switch_points = list(self.switch_points)

    def get_stimuli(self):
        return generate_stimuli_sequence(self)

    def score_experiment(self, dataframe):
        return score_experiment(dataframe, self)


def create_arm_arrays(selfparams):
    """create arm reward probability arrays"""

    switch_points = selfparams.switch_points
    trial_count = selfparams.trial_count
    arm1_base = selfparams.arm1
    arm2_base = selfparams.arm2

    arm1_list = np.array([])
    arm2_list = np.array([])
    block_lengths = np.diff(switch_points, prepend=0, append=trial_count)

    for index, block_length in enumerate(block_lengths):
        if index % 2 == 0:
            arm1, arm2 = arm1_base, arm2_base
        else:
            arm1, arm2 = arm2_base, arm1_base
        arm1_list = np.append(arm1_list, np.repeat(arm1, block_length))
        arm2_list = np.append(arm2_list, np.repeat(arm2, block_length))

    return arm1_list, arm2_list


def generate_stimuli_sequence(taskparams):
    """
    reversal bandit task doesn't have stimuli, so this returns None by default
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

    subject_count = len(df.subject_id.unique())

    # correct score
    arms = np.vstack((taskparams.arm1_list, taskparams.arm2_list))  # row:arm, col:trial
    # arms = [taskparams.arm1, taskparams.arm2]
    # best_arm = np.argmax(arms)
    best_arm = [np.argmax([arms[:, trial]]) for trial in range(taskparams.trial_count)]
    best_arm = np.tile(best_arm, subject_count)
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
