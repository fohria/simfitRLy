import numpy as np
from numba import njit, int32


@njit
def likelihood(parameters, actions, rewards, stimuli):

    alpha = parameters[0]
    beta = parameters[1]
    gamma = parameters[2]

    trial_count = len(actions)
    states = state_representation(stimuli, trial_count, bagsize=3)

    Q = np.ones((9, 3)) * 1 / 3  # 9 states, 3 actions in each

    choice_probabilities = np.zeros(trial_count)

    for trial, (action, reward, state) in enumerate(zip(actions, rewards, states)):

        q_row = Q[state]

        # compute choice probabilities
        q_soft = q_row - np.max(q_row)
        p = np.exp(beta * q_soft) / np.sum(np.exp(beta * q_soft))

        # add choice probability for actual choice
        choice_probabilities[trial] = p[action]

        # can't get next state/observation if we're on last trial
        if trial == trial_count - 1:
            break

        # find maxQ based on next observation
        next_state = states[trial + 1]
        next_qrow = Q[next_state]
        maxQ = np.max(next_qrow)

        # update Q values with reward prediction error (rpe)
        rpe = reward + gamma * maxQ - q_row[action]
        Q[state, action] += alpha * rpe

    # compute and return negative log-likelihood
    loglikelihood = np.sum(np.log(choice_probabilities))
    return -loglikelihood


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
