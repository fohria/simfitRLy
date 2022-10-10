from dataclasses import dataclass
import numpy as np
from numba import njit, int64

from .agent import Agent
from ..utils.choose import choose


@dataclass
class HMMDelta(Agent):
    """
    Hidden Markov Model (HMM) agent, based on Schlagenhauf et al. 2014.

    This version is extended with delta parameter to adjust state transition probability

    TODO: allow changing c and d for other reversal task versions

    This version has two free parameters as per below and their limits in parentheses.

    gamma: the probability that the environment will _stay_ in the current state (0-1)
    delta: the amount gamma will be decreased every trial (0-1)

    An obvious improvement here is to learn the reward probabilities c and d instead of hard coding them based on the task as we do here.
    """

    gamma: float
    delta: float
    name: str = "HMMDelta"

    def play(self, task):
        if task.name == "reversal_bandit":
            return self.play_reversal_bandit(task)
        if task.name == "worthy_bandit":
            return self.play_worthy_bandit(task)
        raise ValueError("woops, that task is not supported!")

    def play_reversal_bandit(self, task):
        stimuli = task.get_stimuli()
        actions, rewards, beliefs = play_reversal_bandit(
            self.gamma,
            self.delta,
            task.trial_count,
            task.arm1_list,
            task.arm2_list,
        )
        return {
            "actions": actions,
            "rewards": rewards,
            "stimuli": stimuli,
            "beliefs": beliefs,
        }

    def play_worthy_bandit(self, task):
        stimuli = task.get_stimuli()
        reward_scale = get_reward_scale(task.reward_scale)
        actions, rewards, beliefs = play_worthy_bandit(
            self.gamma,
            self.delta,
            task.trial_count,
            task.deck1,
            task.deck2,
            reward_scale,
        )
        return {
            "actions": actions,
            "rewards": rewards,
            "stimuli": stimuli,
            "beliefs": beliefs,
        }


def get_reward_scale(rewardtype):
    types = {"normalised": 0, "scaled": 1, "standard": 2}
    return types[rewardtype]


@njit
def play_reversal_bandit(
    gamma: float,
    delta: float,
    trial_count: int,
    arm1_list: np.array,
    arm2_list: np.array,
):

    arm_probs = np.vstack((arm1_list, arm2_list))  # rows: action, columns: trial
    gamma_start = gamma  # use as initial value after a switch
    beta = 20  # softmax inverse temperature. kept constant

    actions = np.zeros(trial_count, dtype=int64)
    rewards = np.zeros(trial_count, dtype=int64)
    beliefs = np.zeros((2, trial_count))  # row: arm, column: trial

    # prior_belief = {1: 0.5, 2: 0.5}  # starting belief is 50/50 for each state
    prior_belief = np.array([0.5, 0.5])  # starting belief is 50/50 for each state

    for timestep in range(trial_count):

        # convert belief into action probabilities
        # probs = np.array([sigmoid(prior_belief[0]), sigmoid(prior_belief[1])])
        probabilities = np.exp(beta * prior_belief) / np.sum(
            np.exp(beta * prior_belief)
        )

        # select action and get reward
        action = choose([0, 1], probabilities=probabilities)
        reward = get_reward(action, timestep, arm_probs)

        # record timestep
        actions[timestep] = action
        rewards[timestep] = reward
        beliefs[:, timestep] = prior_belief

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

        # change gamma every trial
        if np.argmax(prior_belief) == np.argmax(prob_next):
            # stay trial
            newgamma = gamma - delta
            gamma = np.max(np.array([newgamma, 0]))
        else:
            # switch trial, reset switch transition probability
            gamma = gamma_start

        # set the new posterior belief as the current belief (i.e. prior for next step)
        prior_belief = prob_next

    return actions, rewards, beliefs


@njit
def play_worthy_bandit(
    gamma: float,
    delta: float,
    trial_count: int,
    deck1: np.array,
    deck2: np.array,
    reward_scale: int,
):

    decks = np.vstack((deck1, deck2))  # rows: action, columns: trial
    draw_counts = np.zeros(2, dtype=int64)  # action0, action1 draw counts
    gamma_start = gamma  # use as initial value after a switch
    beta = 20  # softmax inverse temperature. kept constant

    actions = np.zeros(trial_count, dtype=int64)
    rewards = np.zeros(trial_count)
    beliefs = np.zeros((2, trial_count))  # row: arm, column: trial

    # prior_belief = {1: 0.5, 2: 0.5}  # starting belief is 50/50 for each state
    prior_belief = np.array([0.5, 0.5])  # starting belief is 50/50 for each state

    for timestep in range(trial_count):

        # convert belief into action probabilities
        # probs = np.array([sigmoid(prior_belief[0]), sigmoid(prior_belief[1])])
        probabilities = np.exp(beta * prior_belief) / np.sum(
            np.exp(beta * prior_belief)
        )

        # select action and get reward
        action = choose([0, 1], probabilities=probabilities)
        reward = decks[action, draw_counts[action]]
        draw_counts[action] += 1

        # record timestep
        actions[timestep] = action
        rewards[timestep] = reward
        beliefs[:, timestep] = prior_belief

        # calculate new belief state based on observation
        nominators = [
            prob_obs_given_state_worthy(action, reward, 0, reward_scale)
            * prior_belief[0],
            prob_obs_given_state_worthy(action, reward, 1, reward_scale)
            * prior_belief[1],
        ]
        denominator = sum(nominators)

        prob_next = np.zeros(2)
        for state_next in [0, 1]:
            prob_next[state_next] = 0
            for state_current in [0, 1]:
                prob_next[state_next] += prob_next_state_given_state(
                    s_next=state_next, s_current=state_current, gamma=gamma
                ) * (nominators[state_current] / denominator)

        # change gamma every trial
        if np.argmax(prior_belief) == np.argmax(prob_next):
            # stay trial
            newgamma = gamma - delta
            gamma = np.max(np.array([newgamma, 0]))
        else:
            # switch trial, reset switch transition probability
            gamma = gamma_start

        # set the new posterior belief as the current belief (i.e. prior for next step)
        prior_belief = prob_next

    return actions, rewards, beliefs


@njit
def sigmoid(x: float):
    """
    make sure probabilities are between 0-1
    """
    return 1 / (1 + np.exp(-20 * (x - 0.5)))


@njit
def get_reward(action: int, trial: int, arms: np.array):
    """
    get reward for the chosen action based on arm reward probabilities in `arms`
    """

    if action == 0:
        if np.random.rand() < arms[0, trial]:  # action, trial
            return 1
        else:
            return 0
    if action == 1:
        if np.random.rand() < arms[1, trial]:  # action, trial
            return 1
        else:
            return 0


@njit
def prob_next_state_given_state(s_next: int, s_current: int, gamma: float = 0.9):
    """
    probability of next state given the current state
    # TODO: make gamma depend on time
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


@njit
def prob_obs_given_state_worthy(
    action: int,
    reward: float,
    s_current: int,
    reward_scale: int,
    c: float = 0.7,
    d: float = 0.7,
):
    """
    probability of observation = (action, reward) given the current state

    because numba does not support scipy we have to create our own normaldist

    however since we have discrete rewards we cheat by having done;
    [norm.cdf(x, loc=5.5, scale=2) for x in range(1, 11)]
    """

    reward_probs = np.array(
        [
            0.012224472655044696,
            0.040059156863817086,
            0.10564977366685535,
            0.2266273523768682,
            0.4012936743170763,
            0.5987063256829237,
            0.7733726476231317,
            0.8943502263331446,
            0.9599408431361829,
            0.9877755273449553,
        ]
    )

    if reward_scale == 0:
        reward = int(denormalise(reward))
    elif reward_scale == 1:
        reward = int(reward * 10)
    else:
        pass  # already have reward scale we need

    reward_index = int64(reward - 1)

    state = s_current

    # in updated version we can have the mean as free parameter
    if action == state:
        return reward_probs[reward_index]
    if action != state:
        return 1 - reward_probs[reward_index]


@njit
def denormalise(xnorm):
    xmin, xmax = 1, 10
    return xnorm * (xmax - xmin) + xmin
