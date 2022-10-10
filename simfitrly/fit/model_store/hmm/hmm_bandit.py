import numpy as np
from numba import njit


@njit
def likelihood(parameters, actions, rewards):

    gamma = parameters[0]

    beta = 20  # softmax inverse temperature. kept constant

    prior_belief = np.array([0.5, 0.5])

    choice_probabilities = np.zeros(len(actions))

    for trial, (action, reward) in enumerate(zip(actions, rewards)):

        # compute choice probabilities
        softvals = prior_belief - np.max(prior_belief)
        p = np.exp(beta * softvals) / np.sum(np.exp(beta * softvals))

        # add choice probability for actual choice
        choice_probabilities[trial] = p[action]

        # calculate new belief state based on observation
        nominators = [
            prob_obs_given_state(action, reward, 0) * prior_belief[0],
            prob_obs_given_state(action, reward, 1) * prior_belief[1],
        ]
        denominator = sum(nominators)

        prob_next = np.zeros(2)
        for state_next in [0, 1]:
            prob_next[state_next] = 0
            for state_current in [0, 1]:
                prob_next[state_next] += prob_next_state_given_state(
                    s_next=state_next, s_current=state_current, gamma=gamma
                ) * (nominators[state_current] / denominator)

        # set the new posterior belief as the current belief (i.e. prior for next step)
        prior_belief = prob_next

    # compute and return negative log-likelihood
    loglikelihood = np.sum(np.log(choice_probabilities))
    return -loglikelihood


@njit
def sigmoid(x: float):
    """
    make sure probabilities are between 0-1
    """
    return 1 / (1 + np.exp(-20 * (x - 0.5)))


@njit
def prob_next_state_given_state(s_next: int, s_current: int, gamma: float = 0.9):
    """
    probability of next state given the current state
    """

    if s_next == s_current:
        return gamma
    else:
        return 1 - gamma


@njit
def prob_obs_given_state(
    action: int, reward: int, s_current: int, c: float = 0.7, d: float = 0.7
):
    """
    probability of observation = (action, reward) given the current state
    """

    state = s_current

    # c and d represent the probabilities of reward/no reward for each arm
    if action == state and reward == 1:
        return 0.5 + 0.5 * c
    if action != state and reward == 1:
        return 0.5 - 0.5 * c

    if action == state and reward == 0:
        return 0.5 - 0.5 * d
    if action != state and reward == 0:
        return 0.5 + 0.5 * d
