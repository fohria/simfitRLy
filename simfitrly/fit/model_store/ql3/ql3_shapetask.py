import numpy as np
from numba import njit


@njit
def likelihood(parameters, actions, rewards, stimuli):

    alpha = parameters[0]
    beta = parameters[1]
    gamma = parameters[2]

    trial_count = len(actions)

    Q = np.repeat(1 / 3, 9).reshape(3, 3)

    choice_probabilities = np.zeros(trial_count)

    for trial, (action, reward, state) in enumerate(zip(actions, rewards, stimuli)):

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
        next_state = stimuli[trial + 1]
        next_qrow = Q[next_state]
        maxQ = np.max(next_qrow)

        # update Q values with reward prediction error (rpe)
        rpe = reward + gamma * maxQ - q_row[action]
        Q[state, action] += alpha * rpe

    # compute and return negative log-likelihood
    loglikelihood = np.sum(np.log(choice_probabilities))
    return -loglikelihood
