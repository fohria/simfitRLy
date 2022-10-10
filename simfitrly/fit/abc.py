import numpy as np
from numba import njit, int32

import simfitrly as sf

from simfitrly.simulate.agents.random_bias import (
    play_shapetask as play_shapetask_randbias,
)
from simfitrly.simulate.agents.ql import play_shapetask as play_shapetask_ql3
from simfitrly.simulate.agents.hrl import play_shapetask as play_shapetask_hrl
from simfitrly.simulate.agents.srtd import play_shapetask as play_shapetask_srtd
from simfitrly.simulate.agents.seql import play_shapetask as play_shapetask_seql
from simfitrly.simulate.agents.seql import state_representation


@njit
def get_scores(actions: np.array, rewards: np.array, stimuli: np.array):

    correct_sums1 = np.sum(rewards[np.arange(0, 99, 3)])
    correct_sums2 = np.sum(rewards[np.arange(1, 99, 3)])
    correct_sums3 = np.sum(rewards[np.arange(2, 99, 3)])
    shiftpred = actions != stimuli
    shift_sums1 = np.sum(shiftpred[np.arange(0, 99, 3)])
    shift_sums2 = np.sum(shiftpred[np.arange(1, 99, 3)])
    shift_sums3 = np.sum(shiftpred[np.arange(2, 99, 3)])

    win = np.append(np.nan, rewards[:-1]) == 1
    stay = np.append(np.nan, actions[:-1]) == actions
    winstay = win & stay
    winstay1 = np.sum(winstay[np.arange(0, 99, 3)])
    winstay2 = np.sum(winstay[np.arange(1, 99, 3)])
    winstay3 = np.sum(winstay[np.arange(2, 99, 3)])
    lose = np.append(np.nan, rewards[:-1]) == 0
    shift = np.append(np.nan, actions[:-1]) != actions
    loseshift = lose & shift
    loseshift1 = np.sum(loseshift[np.arange(0, 99, 3)])
    loseshift2 = np.sum(loseshift[np.arange(1, 99, 3)])
    loseshift3 = np.sum(loseshift[np.arange(2, 99, 3)])

    return np.array(
        [
            correct_sums1,
            correct_sums2,
            correct_sums3,
            winstay1,
            winstay2,
            winstay3,
            loseshift1,
            loseshift2,
            loseshift3,
            shift_sums1,
            shift_sums2,
            shift_sums3,
        ]
    )


@njit
def shapetask_distance(sim, obs):
    """euclidean distance between two coordinates sim and obs"""

    return np.power(np.sum(np.power(np.abs(sim - obs), 2)), 1 / 2)


@njit
def abc_shapetask_randombias(
    bias1: float, bias2: float, trial_count: int, stimuli: np.array
):

    # numba does not support axis=0 for np.mean so we add and then divide
    # sum_scores = np.zeros(12)

    # for i in range(100):
    #     actions, rewards = play_shapetask_randbias(
    #         bias1=bias1, bias2=bias2, trial_count=trial_count, stimuli=stimuli
    #     )
    #     scores = get_scores(actions, rewards, stimuli)
    #     sum_scores = sum_scores + scores

    # return sum_scores / 100

    actions, rewards = play_shapetask_randbias(
        bias1=bias1, bias2=bias2, trial_count=trial_count, stimuli=stimuli
    )
    scores = get_scores(actions, rewards, stimuli)
    return scores


@njit
def abc_shapetask_ql3(
    alpha: float, beta: float, gamma: float, trial_count: int, stimuli: np.array
):

    # sum_scores = np.zeros(12)

    # for i in range(100):
    #     actions, rewards = play_shapetask_ql3(alpha, beta, gamma, trial_count, stimuli)
    #     scores = get_scores(actions, rewards, stimuli)
    #     sum_scores = sum_scores + scores

    # return sum_scores / 100

    actions, rewards = play_shapetask_ql3(alpha, beta, gamma, trial_count, stimuli)
    scores = get_scores(actions, rewards, stimuli)
    return scores


@njit
def abc_shapetask_hrl(
    alpha_low: float,
    alpha_high: float,
    beta_low: float,
    beta_high: float,
    trial_count: int,
    stimuli: np.array,
):

    # sum_scores = np.zeros(12)

    # equivalent to np.tile([0, 0, 1], 33), 99trials/3bagsize
    contexts = np.array([0, 0, 1])
    for _ in range(32):
        contexts = np.append(contexts, [0, 0, 1])

    # for i in range(100):
    #     actions, rewards, tasksets = play_shapetask_hrl(
    #         alpha_low, alpha_high, beta_low, beta_high, trial_count, stimuli, contexts
    #     )
    #     scores = get_scores(actions, rewards, stimuli)
    #     sum_scores = sum_scores + scores

    # return sum_scores / 100

    actions, rewards, tasksets = play_shapetask_hrl(
        alpha_low, alpha_high, beta_low, beta_high, trial_count, stimuli, contexts
    )
    scores = get_scores(actions, rewards, stimuli)
    return scores


@njit
def abc_shapetask_srtd(
    alpha_sr: float,
    alpha_w: float,
    beta: float,
    gamma: float,
    trial_count: int,
    stimuli: np.array,
):

    # sum_scores = np.zeros(12)

    # create states from maze in numba friendly way
    maze = np.arange(9).reshape(3, 3).T
    state_sequence = np.zeros((33, 3), dtype=int32)
    for index, bag in enumerate(stimuli[::3]):
        state_sequence[index, :] = maze[bag]
    state_sequence = np.ravel(state_sequence)

    # for i in range(100):

    #     actions, rewards = play_shapetask_srtd(
    #         alpha_sr,
    #         alpha_w,
    #         beta,
    #         gamma,
    #         trial_count,
    #         stimuli,
    #         state_sequence,
    #     )
    #     scores = get_scores(actions, rewards, stimuli)
    #     sum_scores = sum_scores + scores

    # return sum_scores / 100

    actions, rewards = play_shapetask_srtd(
        alpha_sr,
        alpha_w,
        beta,
        gamma,
        trial_count,
        stimuli,
        state_sequence,
    )
    scores = get_scores(actions, rewards, stimuli)
    return scores


@njit
def abc_shapetask_seql3(
    alpha: float, beta: float, gamma: float, trial_count: int, stimuli: np.array
):

    # sum_scores = np.zeros(12)

    seql_states = state_representation(stimuli, trial_count, 3)  # bagsize=3

    # for i in range(100):

    #     actions, rewards = play_shapetask_seql(
    #         alpha,
    #         beta,
    #         gamma,
    #         trial_count,
    #         seql_states,
    #     )
    #     scores = get_scores(actions, rewards, stimuli)
    #     sum_scores = sum_scores + scores

    # return sum_scores / 100

    actions, rewards = play_shapetask_seql(
        alpha,
        beta,
        gamma,
        trial_count,
        seql_states,
    )
    scores = get_scores(actions, rewards, stimuli)
    return scores
