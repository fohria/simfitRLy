import numpy as np
from numba import njit


@njit
def likelihood(parameters, actions, rewards, stimuli):
    # rewards and stimuli are not used but accepted as inputs to fit with other models

    bias1 = parameters[0]
    bias2 = parameters[1]

    choice_values = np.array([bias1, bias2, 1 - (bias1 + bias2)])
    choice_probabilities = np.zeros(len(actions))

    for trial, action in enumerate(actions):

        # compute choice probabilities
        choice_soft = choice_values - np.max(choice_values)
        p = np.exp(choice_soft) / np.sum(np.exp(choice_soft))

        # add choice probability for actual choice
        choice_probabilities[trial] = p[action]

    # compute and return negative log-likelihood
    loglikelihood = np.sum(np.log(choice_probabilities))
    return -loglikelihood
