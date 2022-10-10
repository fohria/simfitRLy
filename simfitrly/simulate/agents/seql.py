"""
    RL/Sensory prediction behaviour agent with enhanced states.
    a.ka. state enhanced q learning (SEQL)
    this agent will use standard Q-learning to learn responses based on stimuli given.
    HOWEVER: it doesn't use rewards but instead has adapted the RL algorithm to use sensory prediction, i.e. instead of reward, we use the next stimuli

    Q(s, a)_t = Q(s, a)_t + alpha(Q(s | a)_t+1 + gamma * argmaxQ(s_a)_t+1 - Q(s, a)_t)
    where q(s|a)_t+1 is instead of reward
"""

import numpy as np
from dataclasses import dataclass
from numba import njit, int32

from .agent import Agent
from ..utils.choose import choose


@dataclass
class SEQL(Agent):
    """
    define parameter values for the q-learner.
    set gamma = 0 for a simpler form of q-learning (sometimes referred to as 2 parameter q-learning)

    alpha: learning rate
    beta : softmax temperature parameter
    gamma: discount parameter
    """

    alpha: float
    beta: float
    gamma: float
    name: str = "SEQL"

    def play(self, task):
        if task.name == "shapetask":
            return self.play_shapetask(task)
        else:
            raise ValueError("woops, that task is not supported!")

    def play_shapetask(self, taskparams):
        stimuli = taskparams.get_stimuli()
        seql_states = state_representation(
            stimuli, taskparams.trial_count, taskparams.bagsize
        )
        actions, rewards = play_shapetask(
            self.alpha, self.beta, self.gamma, taskparams.trial_count, seql_states
        )
        return {
            "actions": actions,
            "rewards": rewards,
            "stimuli": stimuli,
            "seql_states": seql_states,
        }


@njit
def play_shapetask(
    alpha: float, beta: float, gamma: float, trial_count: int, states: np.array
):
    """
    play standard shapetask with 3 shapes and 3 positions. hardcoded for now.
    """

    qvalues = np.ones((9, 3)) * 1 / 3  # 9 states, 3 actions in each
    actions = np.zeros(trial_count, dtype=int32)
    # rewards = (
    #     np.ones(trial_count, dtype=int32) * -1
    # )  # to signify no reward on last trial
    rewards = np.zeros(trial_count)

    for trial, state in enumerate(states):

        q_row = qvalues[state]

        # softmax to get choice probabilities
        nominator = np.exp(beta * q_row)
        probs = nominator / np.sum(nominator)

        # select action and save it
        action = choose([0, 1, 2], probabilities=probs)
        actions[trial] = action

        # can't get next observation if we're on last trial
        if trial == len(states) - 1:
            break

        # get next state and reward
        next_state = states[trial + 1]
        if action == 0 and next_state in [0, 1, 2]:
            reward = 1
        elif action == 1 and next_state in [3, 4, 5]:
            reward = 1
        elif action == 2 and next_state in [6, 7, 8]:
            reward = 1
        else:
            reward = 0

        rewards[trial] = reward

        next_qrow = qvalues[next_state]
        maxQ = np.max(next_qrow)

        prediction_error = reward + gamma * maxQ - q_row[action]

        qvalues[state, action] += alpha * prediction_error

    return actions, rewards


@njit
def state_representation(sequence, trial_count, bagsize):
    """this will create a state representation for the shapetask sequence that includes shape position. so this representation will use/create position information based on the bagsize."""

    representation_map = np.array(
        [
            [0, 1, 2],  # row0/circle
            [3, 4, 5],  # row1/triangle
            [6, 7, 8],  # row2/square
        ],
        dtype=int32,
    )

    representation_sequence = np.zeros(trial_count, dtype=int32)

    bag_start_indices = np.arange(0, trial_count, bagsize)
    for bag in bag_start_indices:
        seql_states = representation_map[sequence[bag]]
        representation_sequence[bag : bag + 3] = seql_states

    return representation_sequence
