from dataclasses import dataclass
import numpy as np
from numba import njit, int64

from .agent import Agent
from ..utils.choose import choose


@dataclass
class QLDualAlpha(Agent):
    """
    standard Q-learner but separate learning rates for positive/negative prediction errors
    when gamma = 0 we get 2 parameter Q-learning (one step learner)

    alpha_pos: learning rate for positive prediction errors
    alpha_neg: learning rate for negative prediction errors
    beta : softmax temperature parameter
    gamma: discount parameter, defaults to 0
    """

    alpha_pos: float
    alpha_neg: float
    beta: float
    gamma: float = 0
    name: str = "DualAlphaQL"

    def play(self, task):
        if task.name == "2arm_bandit":
            return self.play_bandit(task)
        elif task.name == "reversal_bandit":
            return self.play_reversal_bandit(task)
        elif task.name == "worthy_bandit":
            return self.play_worthy_bandit(task)
        else:
            raise ValueError("woops, that task is not supported!")

    def play_bandit(self, taskparams):
        # stimuli = taskparams.get_stimuli()
        # actions, rewards = ql.play_bandit(
        #     self.alpha,
        #     self.beta,
        #     self.gamma,
        #     taskparams.trials,
        #     taskparams.arm1,
        #     taskparams.arm2,
        # )
        # return actions, rewards, stimuli
        return NotImplementedError(f"regular bandit not implemented for {self.name}")

    def play_reversal_bandit(self, taskparams):
        stimuli = taskparams.get_stimuli()
        actions, rewards = play_reversal_bandit(
            self.alpha_pos,
            self.alpha_neg,
            self.beta,
            self.gamma,
            taskparams.trial_count,
            taskparams.arm1_list,
            taskparams.arm2_list,
        )
        return {"actions": actions, "rewards": rewards, "stimuli": stimuli}

    def play_worthy_bandit(self, taskparams):
        stimuli = taskparams.get_stimuli()
        actions, rewards = play_worthy_bandit(
            self.alpha_pos,
            self.alpha_neg,
            self.beta,
            self.gamma,
            taskparams.trial_count,
            taskparams.deck1,
            taskparams.deck2,
        )
        return {"actions": actions, "rewards": rewards, "stimuli": stimuli}


@njit
def play_reversal_bandit(
    alpha_pos: float,
    alpha_neg: float,
    beta: float,
    gamma: float,
    trial_count: int,
    arm1_list: np.array,
    arm2_list: np.array,
):
    """
    simulate dual q-learner playing 2 armed reversal bandit task

    parameters:
        alpha_pos   : learning rate for positive prediction errors
        alpha_neg   : learning rate for negative prediction errors
        beta        : soft max temperature
        gamma       : future discount
        trial_count : number of bandit arm pulls
        arm1_list   : probability of reward for left arm for each trial
        arm2_list   : probability of reward for right arm for each trial

    returns:
        actions (np.array.int): list of actions
        rewards (np.array.int): list of rewards
    """

    arms = np.vstack((arm1_list, arm2_list))  # rows: action, columns: trial

    actions = np.zeros(trial_count, dtype=int64)
    rewards = np.zeros(trial_count, dtype=int64)

    Q = np.array([0.5, 0.5])  # init with equal probabilities for each action

    for trial in range(trial_count):

        # compute choice probabilities using softmax
        q_soft = Q - np.max(Q)
        probabilities = np.exp(beta * q_soft) / np.sum(np.exp(beta * q_soft))

        # make choice based on choice probabilities
        actions[trial] = choose([0, 1], probabilities)

        # generate reward based on choice
        rewards[trial] = np.random.rand() < arms[actions[trial], trial]

        # update action values
        maxQ = np.max(Q)
        delta = rewards[trial] + gamma * maxQ - Q[actions[trial]]  # prediction error
        if delta >= 0:
            Q[actions[trial]] += alpha_pos * delta
        else:
            Q[actions[trial]] += alpha_neg * delta

    return actions, rewards


@njit
def play_worthy_bandit(
    alpha_pos: float,
    alpha_neg: float,
    beta: float,
    gamma: float,
    trial_count: int,
    deck1: np.array,
    deck2: np.array,
):
    """
    simulate dual q-learner playing 2 armed reversal bandit task

    parameters:
        alpha_pos   : learning rate for positive prediction errors
        alpha_neg   : learning rate for negative prediction errors
        beta        : soft max temperature
        gamma       : future discount
        trial_count : number of bandit arm pulls
        arm1_list   : probability of reward for left arm for each trial
        arm2_list   : probability of reward for right arm for each trial

    returns:
        actions (np.array.int): list of actions
        rewards (np.array.int): list of rewards
    """

    decks = np.vstack((deck1, deck2))  # rows: action, columns: trial
    draw_counts = np.zeros(2, dtype=int64)  # action0, action1 draw counts

    actions = np.zeros(trial_count, dtype=int64)
    rewards = np.zeros(trial_count)

    Q = np.array([0.5, 0.5])

    for trial in range(trial_count):

        # compute choice probabilities using softmax
        q_soft = Q - np.max(Q)
        probabilities = np.exp(beta * q_soft) / np.sum(np.exp(beta * q_soft))

        # make choice based on choice probabilities
        actions[trial] = choose([0, 1], probabilities)

        # generate reward based on choice
        rewards[trial] = decks[actions[trial], draw_counts[actions[trial]]]
        draw_counts[actions[trial]] += 1

        # update action values
        maxQ = np.max(Q)
        delta = rewards[trial] + gamma * maxQ - Q[actions[trial]]  # prediction error
        if delta >= 0:
            Q[actions[trial]] += alpha_pos * delta
        else:
            Q[actions[trial]] += alpha_neg * delta

    return actions, rewards
