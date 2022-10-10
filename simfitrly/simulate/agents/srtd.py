from dataclasses import dataclass
import numpy as np
from numba import njit, int32

from .agent import Agent
from ..utils.choose import choose


@dataclass
class SRTD(Agent):
    """
    SRTD agent based on Russek et al. 2017

    alpha_sr: learning rate for transitions (M)
    alpha_w: learning rate for reward weights (w)
    beta : softmax temperature parameter
    gamma: discount parameter, shared for M and w
    """

    alpha_sr: float
    alpha_w: float
    beta: float
    gamma: float
    name: str = "SRTD"

    def play(self, task):
        if task.name == "shapetask":
            return self.play_shapetask(task)
        else:
            raise ValueError("woops, that task is not supported!")

    def play_shapetask(self, taskparams):
        # note: maze is hardcoded to 3shapes, 3positions for now
        stimuli = taskparams.get_stimuli()
        maze = np.arange(9).reshape(3, 3).T
        state_sequence = [maze[bag] for bag in stimuli[:: taskparams.bagsize]]
        state_sequence = np.ravel(state_sequence)

        actions, rewards = play_shapetask(
            self.alpha_sr,
            self.alpha_w,
            self.beta,
            self.gamma,
            taskparams.trial_count,
            stimuli,
            state_sequence,
        )

        return {"actions": actions, "rewards": rewards, "stimuli": stimuli}


@njit
def play_shapetask(
    sr_alpha: float,
    w_alpha: float,
    beta: float,
    gamma: float,
    trial_count: int,
    shape_sequence: np.array,
    state_sequence: np.array,
):

    M = np.identity(27)  # maybe can get the 27 from var/values later
    w = np.zeros((27, 1))
    next_states = np.arange(27).reshape(9, 3)

    # storage (start as -1 to easier see what was updated in loop)
    # actions = np.ones(trial_count, dtype=int32) * -1
    # rewards = np.ones(trial_count) * -1
    actions = np.zeros(trial_count, dtype=int32)
    rewards = np.zeros(trial_count)

    # take first action, based on starting state
    state_prime = state_sequence[0]
    action_prime = srtd_eval(M, w, state_prime, next_states, beta)

    # for step in range(trial_count):
    for step in range(trial_count):

        # set current state, action to previous trial's next/prime
        state = state_prime
        action = action_prime
        actions[step] = action

        # can't get next state if we're on last trial
        if step == trial_count - 1:
            break

        # get and save reward
        if action == shape_sequence[step + 1]:
            reward = 1
        else:
            reward = 0
        rewards[step] = reward

        state_prime = state_sequence[step + 1]

        # eval/get action
        action_prime = srtd_eval(M, w, state_prime, next_states, beta)

        # get M/w compound states for updates
        Mstate = next_states[state, action]
        Mstate_prime = next_states[state_prime, action_prime]

        # update M
        M = srtd_updateM(M, sr_alpha, gamma, Mstate, Mstate_prime)

        # update w
        w = srtd_update_w(w, w_alpha, gamma, Mstate, Mstate_prime, reward, M)

    return actions, rewards


@njit
def srtd_eval(M, w, state, next_states, beta):

    # get qvalues from V
    V = M @ w
    next_state = next_states[state]
    q_values = np.ravel(V[next_state])

    # calculate action probabilities with softmax
    exp_qvals = np.exp(beta * q_values)
    action_probs = exp_qvals / np.sum(exp_qvals)

    action = choose([0, 1, 2], action_probs)

    return action


@njit
def srtd_updateM(M, sr_alpha, gamma, state, state_prime):

    t = np.zeros(len(M))
    t[state] = 1

    M[state, :] = (1 - sr_alpha) * M[state, :] + sr_alpha * (
        t + gamma * M[state_prime, :]
    )

    return M


@njit
def srtd_update_w(w, w_alpha, gamma, state, state_prime, reward, M):

    feature_rep_s = M[state, :].reshape((1, len(M)))
    norm_feature_rep_s = feature_rep_s / (feature_rep_s @ feature_rep_s.T)

    w_error = reward + gamma * (M[state_prime, :] @ w) - (M[state, :] @ w)
    w = w + w_alpha * w_error * norm_feature_rep_s.T

    return w
