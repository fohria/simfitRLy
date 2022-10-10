from dataclasses import dataclass
import numpy as np
from numba import njit, int64

from .agent import Agent
from ..utils.choose import choose


@dataclass
class QLDualUpdate(Agent):
    """
    Q-learner where both action values are updated every trial

    when gamma = 0 we get 2 parameter Q-learning (one step learner)

    alpha: learning rate
    beta : softmax temperature parameter
    gamma: discount parameter, defaults to 0
    """

    alpha: float
    beta: float
    gamma: float = 0
    name: str = "DualUpdateQL"

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
        return NotImplementedError(f"regular bandit not implemented for {self.name}")

    def play_reversal_bandit(self, taskparams):
        stimuli = taskparams.get_stimuli()
        actions, rewards = play_reversal_bandit(
            self.alpha,
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
            self.alpha,
            self.beta,
            self.gamma,
            taskparams.trial_count,
            taskparams.deck1,
            taskparams.deck2,
        )
        return {"actions": actions, "rewards": rewards, "stimuli": stimuli}


@njit
def play_reversal_bandit(
    alpha: float,
    beta: float,
    gamma: float,
    trial_count: int,
    arm1_list: np.array,
    arm2_list: np.array,
):
    """
    simulate dual update q-learner playing 2 armed reversal bandit task

    parameters:
        alpha       : learning rate
        beta        : softmax temperature
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
        delta_chosen = rewards[trial] + gamma * maxQ - Q[actions[trial]]
        delta_unchosen = -rewards[trial] + gamma * maxQ - Q[1 - actions[trial]]

        Q[actions[trial]] += alpha * delta_chosen
        Q[1 - actions[trial]] += alpha * delta_unchosen

    return actions, rewards


@njit
def play_worthy_bandit(
    alpha: float,
    beta: float,
    gamma: float,
    trial_count: int,
    deck1: np.array,
    deck2: np.array,
):
    """
    simulate dual update q-learner playing 2 armed reversal bandit task

    parameters:
        alpha       : learning rate
        beta        : softmax temperature
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

    Q = np.array([0.5, 0.5])  # init with equal probabilities for each action

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
        delta_chosen = rewards[trial] + gamma * maxQ - Q[actions[trial]]
        delta_unchosen = -rewards[trial] + gamma * maxQ - Q[1 - actions[trial]]

        Q[actions[trial]] += alpha * delta_chosen
        Q[1 - actions[trial]] += alpha * delta_unchosen

    return actions, rewards
