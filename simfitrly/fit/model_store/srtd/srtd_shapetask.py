import numpy as np
from numba import njit, int32


@njit
def likelihood(parameters, actions, rewards, states):

    alpha_sr = parameters[0]
    alpha_w = parameters[1]
    beta = parameters[2]
    gamma = parameters[3]

    trial_count = len(actions)

    M = np.identity(27)
    w = np.zeros((27, 1))
    next_states = np.arange(27).reshape(9, 3)

    choice_probabilities = np.zeros(trial_count)

    # take first action, based on starting state
    state_prime = states[0]
    # action_prime = srtd_eval(M, w, state_prime, next_states, beta)
    action_probs = srtd_eval(M, w, state_prime, next_states, beta)

    for step, (action, reward, state) in enumerate(zip(actions, rewards, states)):

        # save probability of the action taken
        state = state_prime
        # action = action_prime
        # actions[step] = action
        choice_probabilities[step] = action_probs[action]

        # can't get next state if we're on last trial
        if step == trial_count - 1:
            break

        state_prime = states[step + 1]

        # eval/get action
        # action_prime = srtd_eval(M, w, state_prime, next_states, beta)
        action_probs = srtd_eval(M, w, state_prime, next_states, beta)
        action_prime = actions[step + 1]

        # get M/w compound states for updates
        Mstate = next_states[state, action]
        Mstate_prime = next_states[state_prime, action_prime]

        # update M
        M = srtd_updateM(M, alpha_sr, gamma, Mstate, Mstate_prime)

        # update w
        w = srtd_update_w(w, alpha_w, gamma, Mstate, Mstate_prime, reward, M)

    # compute and return negative log-likelihood
    loglikelihood = np.sum(np.log(choice_probabilities))
    return -loglikelihood


@njit
def srtd_eval(M, w, state, next_states, beta):

    # get qvalues from V
    V = M @ w
    next_state = next_states[state]
    q_values = np.ravel(V[next_state])

    # calculate action probabilities with softmax
    exp_qvals = np.exp(beta * q_values)
    action_probs = exp_qvals / np.sum(exp_qvals)

    # action = choose([0, 1, 2], action_probs)

    return action_probs


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
