import numpy as np
from numba import njit


@njit
def likelihood(parameters, actions, rewards):

    alpha_pos = parameters[0]
    alpha_neg = parameters[1]
    beta = parameters[2]

    Q = np.array([0.5, 0.5])

    choice_probabilities = np.zeros(len(actions))

    for trial, (action, reward) in enumerate(zip(actions, rewards)):

        # compute choice probabilities
        q_soft = Q - np.max(Q)
        p = np.exp(beta * q_soft) / np.sum(np.exp(beta * q_soft))

        # add choice probability for actual choice
        choice_probabilities[trial] = p[action]

        # update action values
        delta = reward - Q[action]
        if delta >= 0:
            Q[action] += alpha_pos * delta
        else:
            Q[action] += alpha_neg * delta

    # compute and return negative log-likelihood
    loglikelihood = np.sum(np.log(choice_probabilities))
    return -loglikelihood
