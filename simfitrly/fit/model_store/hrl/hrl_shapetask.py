import numpy as np
from numba import njit, int32


@njit
def likelihood(parameters, actions, rewards, stimuli, tasksets):
    """
    context used is 'lastinbag' from simulated agent
    for now, number of shapes/actions etc are hardcoded to 3shapes, bagsize3
    """

    alpha_low = parameters[0]
    alpha_high = parameters[1]
    beta_low = parameters[2]
    beta_high = parameters[3]

    bagsize = 3
    trial_count = len(actions)

    contexts = np.array([[0, 0, 1] for _ in range(trial_count // bagsize)], dtype=int32)
    contexts = np.ravel(contexts)

    q_high = np.ones((2, 2)) * 0.5  # 2 task sets: one for each context
    q_low = np.ones((2, 3, 3)) * 1 / 3  # 2tasksets, 3 shapes, 3 actions

    choice_probabilities = np.zeros((trial_count, 2))

    for trial, (action, reward, context, state, taskset) in enumerate(
        zip(actions, rewards, contexts, stimuli, tasksets)
    ):

        # select task set based on context and q_high values
        q_high_row = q_high[context]
        p_high = softmax(beta_high * q_high_row)
        # taskset = choose([0, 1], p_high)
        choice_probabilities[trial, 0] = p_high[taskset]

        # select action based on q_low values for selected task set
        q_low_row = q_low[taskset, state]
        p_low = softmax(beta_low * q_low_row)

        # save probability of the action that was chosen
        choice_probabilities[trial, 1] = p_low[action]

        # there are no rewards for last trial, as it depends on next stimuli
        if trial == trial_count - 1:
            break

        # calculate reward prediction error for low values
        rpe_low = reward - q_low_row[action]
        q_low[taskset, state, action] += alpha_low * rpe_low

        # calculate reward prediction error for high values
        rpe_high = reward - q_high_row[taskset]
        q_high[context, taskset] += alpha_high * rpe_high

    # compute and return negative log-likelihood
    loglikelihood = np.sum(np.log(choice_probabilities))
    return -loglikelihood


@njit
def softmax(array):
    exp_qvals = np.exp(array)
    probs = exp_qvals / np.sum(exp_qvals)
    return probs


# numba doesn't understand numpy's random choice function
# https://github.com/numba/numba/issues/2539#issuecomment-507306369
@njit
def choose(array, probabilities):
    """
    :param array: A 1D numpy array of values to sample from.
    :param probabilities: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return array[
        np.searchsorted(np.cumsum(probabilities), np.random.random(), side="right")
    ]
