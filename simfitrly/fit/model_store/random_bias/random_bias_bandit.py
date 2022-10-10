import numpy as np
from numba import njit


@njit
def likelihood(parameters, actions, rewards):

    bias = parameters[0]
    choice_values = np.array([bias, 1-bias])
    choice_probabilities = np.zeros(len(actions))

    for trial, (action, reward) in enumerate(zip(actions, rewards)):

        # compute choice probabilities
        choice_soft = choice_values - np.max(choice_values)
        p = np.exp(choice_soft) / np.sum(np.exp(choice_soft))

        # add choice probability for actual choice
        choice_probabilities[trial] = p[action]

    # compute and return negative log-likelihood
    loglikelihood = np.sum(np.log(choice_probabilities))
    return -loglikelihood
