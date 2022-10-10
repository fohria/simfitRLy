import numpy as np
from numba import njit


@njit
def likelihood(parameters, actions, rewards, stimuli):

    alpha = parameters[0]
    beta = parameters[1]

    Q = np.ones((2, 2)) * 0.5  # 2 states, 2 actions (row, column)

    choice_probabilities = np.zeros(len(actions))

    for trial, (action, reward, state) in enumerate(zip(actions, rewards, stimuli)):

        q_row = Q[state]

        # compute choice probabilities
        q_soft = q_row - np.max(q_row)
        p = np.exp(beta * q_soft) / np.sum(np.exp(beta * q_soft))

        # add choice probability for actual choice
        choice_probabilities[trial] = p[action]

        # update values
        delta = reward - q_row[action]
        Q[state, action] += alpha * delta

    # compute and return negative log-likelihood
    loglikelihood = np.sum(np.log(choice_probabilities))
    return -loglikelihood
