from dataclasses import dataclass
import numpy as np
from numba import njit, int64, int32

from .agent import Agent
from ..utils.choose import choose


@dataclass
class RandomBias(Agent):
    """
    this agent selects actions randomly. it can have a bias towards certain actions.

    bias: bias to select one action over another

    TODO: if we know we will have max 4 actions, then we have 3 bias params
    but 2 are optional so basic form is 2 actions with one bias

    """

    bias1: float
    bias2: float = 0  # for 3 actions like shapetask
    name: str = "RandomBias"

    def __post_init__(self):
        if self.bias2 != 0:  # assume bias2 not used if it's at default value
            self.bias2 = np.random.uniform(0, 1 - self.bias1)

    def play(self, taskparams):
        if taskparams.name == "2arm_bandit":
            return self.play_bandit(taskparams)
        if taskparams.name == "reversal_bandit":
            return self.play_reversal_bandit(taskparams)
        if taskparams.name == "worthy_bandit":
            return self.play_worthy_bandit(taskparams)
        if taskparams.name == "shapetask":
            return self.play_shapetask(taskparams)
        raise ValueError("this task is unknown to RandomBias agent!")

    def play_bandit(self, taskparams):
        stimuli = taskparams.get_stimuli()
        actions, rewards = play_bandit(
            self.bias1,
            taskparams.trial_count,
            taskparams.arm1,
            taskparams.arm2,
        )
        return {"actions": actions, "rewards": rewards, "stimuli": stimuli}

    def play_reversal_bandit(self, taskparams):
        stimuli = taskparams.get_stimuli()
        actions, rewards = play_reversal_bandit(
            self.bias1,
            taskparams.trial_count,
            taskparams.arm1_list,
            taskparams.arm2_list,
        )
        return {"actions": actions, "rewards": rewards, "stimuli": stimuli}

    def play_worthy_bandit(self, taskparams):
        stimuli = taskparams.get_stimuli()
        actions, rewards = play_worthy_bandit(
            self.bias1,
            taskparams.trial_count,
            taskparams.deck1,
            taskparams.deck2,
        )
        return {"actions": actions, "rewards": rewards, "stimuli": stimuli}

    def play_shapetask(self, taskparams):
        stimuli = taskparams.get_stimuli()
        actions, rewards = play_shapetask(
            self.bias1,
            self.bias2,
            taskparams.trial_count,
            stimuli,
        )
        return {"actions": actions, "rewards": rewards, "stimuli": stimuli}


@njit
def play_bandit(bias: float, trial_count: int, arm1: float, arm2: float):
    """
    simulate an agent taking random actions in bandit task
    it may have a bias towards one arm over the other

    parameters:
        bias: tendency to pick one arm over the other
        trial_count : number of bandit arm pulls
        arm1: probability of reward for left arm
        arm2: probability of reward for right arm

    returns:
        actions (np.array.int): list of actions
        rewards (np.array.int): list of rewards
    """

    bandit = np.array([arm1, arm2])

    actions = np.zeros(trial_count, dtype=int64)
    rewards = np.zeros(trial_count, dtype=int64)

    choice_probs = np.array([bias, 1 - bias])

    for trial in range(trial_count):

        # make choice based on choice probabilities
        actions[trial] = choose([0, 1], choice_probs)

        # generate reward based on choice
        rewards[trial] = np.random.rand() < bandit[actions[trial]]

    return actions, rewards


@njit
def play_reversal_bandit(
    bias: float,
    trial_count: int,
    arm1_list: np.array,
    arm2_list: np.array,
):
    """
    simulate playing 2 armed reversal bandit task selecting actions at random

    parameters:
        bias: tendency to pick one arm over the other
        trial_count : number of bandit arm pulls
        arm1_list: probability of reward for left arm for each trial
        arm2_list: probability of reward for right arm for each trial

    returns:
        actions (np.array.int): list of actions
        rewards (np.array.int): list of rewards
    """

    arms = np.vstack((arm1_list, arm2_list))  # rows: action, columns: trial

    actions = np.zeros(trial_count, dtype=int64)
    rewards = np.zeros(trial_count, dtype=int64)

    choice_probs = np.array([bias, 1 - bias])  # probabilities are same for all trials

    for trial in range(trial_count):

        # make choice based on choice probabilities
        actions[trial] = choose([0, 1], choice_probs)

        # generate reward based on choice
        rewards[trial] = np.random.rand() < arms[actions[trial], trial]

    return actions, rewards


@njit
def play_worthy_bandit(
    bias: float,
    trial_count: int,
    deck1: np.array,
    deck2: np.array,
):
    """
    simulate playing worthy bandit task selecting actions at random

    parameters:
        bias: tendency to pick one arm over the other
        trial_count : number of bandit arm pulls
        deck1: rewards for cards in deck1
        deck2: rewards for cards in deck2

    returns:
        actions (np.array.int): list of actions
        rewards (np.array.int): list of rewards
    """

    decks = np.vstack((deck1, deck2))  # rows: action, columns: deck position
    draw_counts = np.zeros(2, dtype=int64)

    actions = np.zeros(trial_count, dtype=int64)
    rewards = np.zeros(trial_count)

    choice_probs = np.array([bias, 1 - bias])  # probabilities are same for all trials

    for trial in range(trial_count):

        # make choice based on choice probabilities
        actions[trial] = choose([0, 1], choice_probs)

        # generate reward based on choice
        rewards[trial] = decks[actions[trial], draw_counts[actions[trial]]]
        draw_counts[actions[trial]] += 1

    return actions, rewards


@njit
def play_shapetask(bias1: float, bias2: float, trial_count: int, stimuli: np.array):

    probabilities = np.array([bias1, bias2, 1 - (bias1 + bias2)])

    actions = np.zeros(trial_count, dtype=int32)
    rewards = np.zeros(trial_count, dtype=int32)

    for trial in range(trial_count):
        action = choose(np.array([0, 1, 2]), probabilities)
        actions[trial] = action

        # can't get next state/observation if we're on last trial
        if trial == trial_count - 1:
            break

        if action == stimuli[trial + 1]:
            reward = 1
        else:
            reward = 0
        rewards[trial] = reward

    return actions, rewards
