"""
    reversal bandit task.
    in this task, there are no stimuli, only 2 arms to pull.
    at switch points, the arm reward probabilities change
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .task import Task


@dataclass
class WorthyBandit(Task):
    """

    variant of probabilistic reversal bandit task as per Worthy et al. 2007

    2 armed reversal bandit task. on creation, arm1 and arm2 will be combined with
    switch_points to create additional arm1, arm2 arrays for entire trial_count length.
    these arrrays are called arm1_list and arm2_list respectively and are useful when
    simulating the task.

        deck1        : list of rewards for each card in deck1
        deck2        : list of rewards for each card in deck2
        trial_count  : total number of trials
        reward_scale : what scale rewards should be, it has three settings:
             standard  : rewards are 1-10 as for the human participants (default)
             normalised: rewards are between 0 and 1, with 1=0 and 10=1
             scaled    : rewards are between 0.1 and 1 (reward / 10)
        name         : name of the task (default is worthy_bandit)
    """

    deck1: list = field(default_factory=lambda: get_deck(1))
    deck2: list = field(default_factory=lambda: get_deck(2))
    trial_count: int = 80
    reward_scale: str = "standard"
    name: str = "worthy_bandit"

    def __post_init__(self):
        if self.reward_scale == "normalised":
            self.deck1 = np.array([normalise(r) for r in self.deck1])
            self.deck2 = np.array([normalise(r) for r in self.deck2])
        if self.reward_scale == "scaled":
            self.deck1 = np.array([r / 10 for r in self.deck1])
            self.deck2 = np.array([r / 10 for r in self.deck2])

    def get_stimuli(self):
        return generate_stimuli_sequence(self)

    def score_experiment(self, dataframe):
        return score_experiment(dataframe, self)


def normalise(x):
    # we know max=10 and min=1
    x_max, x_min = 10, 1
    return (x - x_min) / (x_max - x_min)


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
        rewardcumsum - running total of reward
        rewardsum - reward at the end of task
        correct - means picking the deck with highest mean
        good_draws - sum of cards picked from good deck
    """

    # subject_count = len(df.subject_id.unique())

    df["rewardcumsum"] = 0
    df["rewardsum"] = 0
    df["correct"] = 0
    df["good_draws_cum"] = 0
    df["good_draws"] = 0
    df["real_reward"] = 0
    df["probdeck1toend"] = 0

    for subject in df.subject_id.unique():

        subdata = df.query("subject_id == @subject")

        # transform reward values as needed depending on task config
        if taskparams.reward_scale == "standard":
            rewards = subdata.reward.to_numpy()
        elif taskparams.reward_scale == "normalised":
            rewards = np.array([denormalise(r) for r in subdata.reward])
        elif taskparams.reward_scale == "scaled":
            rewards = np.array([r * 10 for r in subdata.reward])
        else:
            raise Warning("wrong value for reward_scale, check data!")
        df.loc[df.subject_id == subject, "real_reward"] = rewards

        # reward sums
        cumsum = np.cumsum(rewards)
        df.loc[df.subject_id == subject, "rewardcumsum"] = cumsum
        df.loc[df.subject_id == subject, "rewardsum"] = np.tile(
            cumsum[-1], taskparams.trial_count
        )

        # correct score
        # deck1 = good deck, deck2 = bad deck
        # actions will be 0/1 so deck1 = 0 and deck2 = 1 below
        sub_correct = get_subject_correct(subdata.action)
        df.loc[df.subject_id == subject, "correct"] = sub_correct

        # number of cards drawn from good deck
        good_draws = np.cumsum(subdata.action == 0)
        df.loc[df.subject_id == subject, "good_draws_cum"] = good_draws
        df.loc[df.subject_id == subject, "good_draws"] = np.tile(
            good_draws.to_numpy()[-1], taskparams.trial_count
        )

        # probability of deck1 from current trial to end
        probdeck1 = get_probdeck1toend(subdata.action.to_numpy())
        df.loc[df.subject_id == subject, "probdeck1toend"] = probdeck1

    return df


def get_probdeck1toend(actions):
    actions_rev = np.flip(actions)
    actions_rev_cum = np.cumsum(actions_rev == 0)
    cum_rev = np.flip(actions_rev_cum)
    pdeck1 = cum_rev / np.flip(np.arange(1, 81))
    return pdeck1


def denormalise(xnorm):
    xmin, xmax = 1, 10
    return xnorm * (xmax - xmin) + xmin


def get_subject_correct(actions):
    gd_count = 0
    bd_count = 0
    correct = []
    for action in actions:
        if action == 0:
            gd_count += 1
        if action == 1:
            bd_count += 1
        state = get_state(gd_count, bd_count)
        if action == state:
            correct.append(1)
        else:
            correct.append(0)

    return np.array(correct)


def get_state(GD_count, BD_count):
    if BD_count < 30:
        state = 1  # BD best
    else:
        if BD_count > 50:
            state = 0  # GD best
        else:
            if GD_count < 20:
                state = 1
            else:
                state = 0
    return state


def get_deck(deck):
    if deck == 1:
        return np.array(
            [
                2,
                2,
                1,
                1,
                2,
                1,
                1,
                3,
                2,
                6,
                2,
                8,
                1,
                6,
                2,
                1,
                1,
                5,
                8,
                5,
                10,
                10,
                8,
                3,
                10,
                7,
                10,
                8,
                3,
                4,
                9,
                10,
                3,
                6,
                3,
                5,
                10,
                10,
                10,
                7,
                3,
                8,
                5,
                8,
                6,
                9,
                4,
                4,
                4,
                10,
                6,
                4,
                10,
                3,
                10,
                5,
                10,
                3,
                10,
                10,
                5,
                4,
                6,
                10,
                7,
                7,
                10,
                10,
                10,
                3,
                1,
                4,
                1,
                3,
                1,
                7,
                1,
                3,
                1,
                8,
            ]
        )
    if deck == 2:
        return np.array(
            [
                7,
                10,
                5,
                10,
                6,
                6,
                10,
                10,
                10,
                8,
                4,
                8,
                10,
                4,
                9,
                10,
                8,
                6,
                10,
                10,
                10,
                4,
                7,
                10,
                5,
                10,
                4,
                10,
                10,
                9,
                2,
                9,
                8,
                10,
                7,
                7,
                1,
                10,
                2,
                6,
                4,
                7,
                2,
                1,
                1,
                1,
                7,
                10,
                1,
                4,
                2,
                1,
                1,
                1,
                4,
                1,
                4,
                1,
                1,
                1,
                1,
                3,
                1,
                4,
                1,
                1,
                1,
                5,
                1,
                1,
                1,
                7,
                2,
                1,
                2,
                1,
                4,
                1,
                4,
                1,
            ]
        )
    return ValueError("no such deck!")
