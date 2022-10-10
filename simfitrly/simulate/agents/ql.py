from dataclasses import dataclass
import numpy as np
from numba import njit, int64

from .agent import Agent
from ..utils.choose import choose


@dataclass
class QL(Agent):
    """
    standard Q-learner a la Watkins
    when gamma = 0 we get 2 parameter Q-learning (one step learner)

    alpha: learning rate
    beta : softmax temperature parameter
    gamma: discount parameter, defaults to 0
    """

    alpha: float
    beta: float
    gamma: float = 0
    name: str = "QL"

    def play(self, task):
        # TODO: we could potentially make even nicer than if statements but whatever
        if task.name == "2arm_bandit":
            return self.play_bandit(task)
        elif task.name == "reversal_bandit":
            return self.play_reversal_bandit(task)
        elif task.name == "worthy_bandit":
            return self.play_worthy_bandit(task)
        elif task.name == "shapetask":
            return self.play_shapetask(task)
        else:
            raise ValueError("woops, that task is not supported!")

    def play_bandit(self, taskparams):
        stimuli = taskparams.get_stimuli()
        actions, rewards = play_bandit(
            self.alpha,
            self.beta,
            self.gamma,
            taskparams.trial_count,
            taskparams.arm1,
            taskparams.arm2,
        )
        return {"actions": actions, "rewards": rewards, "stimuli": stimuli}

    def play_reversal_bandit(self, taskparams):
        stimuli = taskparams.get_stimuli()
        actions, rewards = play_reversal_bandit(
            self.alpha,
            self.beta,
            self.gamma,
            taskparams.trial_count,
            taskparams.arm1_list,
            taskparams.arm2_list,
        )
        return {"actions": actions, "rewards": rewards, "stimuli": stimuli}

    def play_reversal_bandit_states(self, taskparams):
        stimuli = [0 if x == taskparams.arm1 else 1 for x in taskparams.arm1_list]
        stimuli = np.array(stimuli)
        actions, rewards = play_reversal_bandit_states(
            self.alpha,
            self.beta,
            self.gamma,
            taskparams.trial_count,
            taskparams.arm1_list,
            taskparams.arm2_list,
            stimuli,
        )
        return {"actions": actions, "rewards": rewards, "stimuli": stimuli}

    def play_worthy_bandit(self, taskparams):
        stimuli = taskparams.get_stimuli()
        actions, rewards = play_worthy_bandit(
            self.alpha,
            self.beta,
            self.gamma,
            taskparams.trial_count,
            taskparams.deck1,
            taskparams.deck2,
        )
        return {"actions": actions, "rewards": rewards, "stimuli": stimuli}

    def play_worthy_bandit_states(self, taskparams):
        actions, rewards, stimuli = play_worthy_bandit_states(
            self.alpha,
            self.beta,
            self.gamma,
            taskparams.trial_count,
            taskparams.deck1,
            taskparams.deck2,
        )
        return {"actions": actions, "rewards": rewards, "stimuli": stimuli}

    def play_shapetask(self, taskparams):
        stimuli = taskparams.get_stimuli()
        actions, rewards = play_shapetask(
            self.alpha, self.beta, self.gamma, taskparams.trial_count, stimuli
        )
        return {"actions": actions, "rewards": rewards, "stimuli": stimuli}


@njit
def play_bandit(
    alpha: float, beta: float, gamma: float, trial_count: int, arm1: float, arm2: float
):
    """
    simulate q-learner playing 2 armed bandit task

    parameters:
        alpha: learning rate
        beta : soft max temperature
        gamma: future discount
        trial_count : number of bandit arm pulls
        arm1: probability of reward for left arm
        arm2: probability of reward for right arm

    returns:
        actions (np.array.int): list of actions
        rewards (np.array.int): list of rewards
        q_history: list of lists for q value history
    """

    bandit = np.array([arm1, arm2])

    actions = np.zeros(trial_count, dtype=int64)
    rewards = np.zeros(trial_count, dtype=int64)

    Q = np.array([0.5, 0.5])  # init with equal probabilities for each action

    for trial in range(trial_count):

        # compute choice probabilities using softmax
        q_soft = Q - np.max(Q)
        probabilities = np.exp(beta * q_soft) / np.sum(np.exp(beta * q_soft))

        # make choice based on choice probabilities
        actions[trial] = choose([0, 1], probabilities)

        # generate reward based on choice
        rewards[trial] = np.random.rand() < bandit[actions[trial]]

        # update action values
        maxQ = np.max(Q)
        delta = rewards[trial] + gamma * maxQ - Q[actions[trial]]  # prediction error
        Q[actions[trial]] += alpha * delta

    return actions, rewards


@njit
def play_reversal_bandit(
    alpha: float,
    beta: float,
    gamma: float,
    trial_count: int,
    arm1_list: np.array,
    arm2_list: np.array,
):
    """
    simulate q-learner playing 2 armed reversal bandit task

    parameters:
        alpha: learning rate
        beta : soft max temperature
        gamma: future discount
        trial_count : number of bandit arm pulls
        arm1_list: probability of reward for left arm for each trial
        arm2_list: probability of reward for right arm for each trial

    returns:
        actions (np.array.int): list of actions
        rewards (np.array.int): list of rewards
        q_history: list of lists for q value history
    """

    if len(arm1_list) != trial_count or len(arm2_list) != trial_count:
        # numba doesn't support f-strings fully yet
        print("arm1list length: " + str(len(arm1_list)))
        print("arm2list length: " + str(len(arm2_list)))
        print("trialcount: " + str(trial_count))
        raise ValueError("length of inputs do not match!")

    arms = np.vstack((arm1_list, arm2_list))  # rows: action, columns: trial

    actions = np.zeros(trial_count, dtype=int64)
    rewards = np.zeros(trial_count, dtype=int64)

    Q = np.array([0.5, 0.5])  # init with equal probabilities for each action

    for trial in range(trial_count):

        # compute choice probabilities using softmax
        q_soft = Q - np.max(Q)
        probabilities = np.exp(beta * q_soft) / np.sum(np.exp(beta * q_soft))

        # make choice based on choice probabilities
        actions[trial] = choose([0, 1], probabilities)

        # generate reward based on choice
        rewards[trial] = np.random.rand() < arms[actions[trial], trial]

        # update action values
        maxQ = np.max(Q)
        delta = rewards[trial] + gamma * maxQ - Q[actions[trial]]  # prediction error
        Q[actions[trial]] += alpha * delta

    return actions, rewards


@njit
def play_reversal_bandit_states(
    alpha: float,
    beta: float,
    gamma: float,
    trial_count: int,
    arm1_list: np.array,
    arm2_list: np.array,
    stimuli: np.array,
):
    """
    simulate q-learner playing 2 armed reversal bandit task

    parameters:
        alpha: learning rate
        beta : soft max temperature
        gamma: future discount
        trial_count : number of bandit arm pulls
        arm1_list: probability of reward for left arm for each trial
        arm2_list: probability of reward for right arm for each trial
        stimuli: list of what state we are in on each trial

    returns:
        actions (np.array.int): list of actions
        rewards (np.array.int): list of rewards
        q_history: list of lists for q value history
    """

    arms = np.vstack((arm1_list, arm2_list))  # rows: action, columns: trial

    actions = np.zeros(trial_count, dtype=int64)
    rewards = np.zeros(trial_count, dtype=int64)

    # init with equal probabilities for each action
    Q = np.ones((2, 2)) * 0.5  # 2 states, 2 actions (row, column)

    for trial, observation in enumerate(stimuli):

        q_row = Q[observation]

        # compute choice probabilities using softmax
        q_soft = q_row - np.max(q_row)  # trick for numerical stability
        probabilities = np.exp(beta * q_soft) / np.sum(np.exp(beta * q_soft))

        # make choice based on choice probabilities
        actions[trial] = choose([0, 1], probabilities)

        # generate reward based on choice
        rewards[trial] = np.random.rand() < arms[actions[trial], trial]

        # cant update on last trial as there is no next observation
        if trial < trial_count - 1:
            # find maxQ based on next observation
            next_observation = stimuli[trial + 1]
            next_qrow = Q[next_observation]
            maxQ = np.max(next_qrow)

            # update action values
            delta = (
                rewards[trial] + gamma * maxQ - q_row[actions[trial]]
            )  # prediction error
            Q[observation, actions[trial]] += alpha * delta

    return actions, rewards


@njit
def play_worthy_bandit(
    alpha: float,
    beta: float,
    gamma: float,
    trial_count: int,
    deck1: np.array,
    deck2: np.array,
):
    """
    simulate q-learner playing worthy bandit task

    parameters:
        alpha: learning rate
        beta : soft max temperature
        gamma: future discount
        trial_count : number of bandit arm pulls
        deck1: probability of reward for left arm for each trial
        deck2: probability of reward for right arm for each trial

    returns:
        actions (np.array.int): list of actions
        rewards (np.array.int): list of rewards
    """

    decks = np.vstack((deck1, deck2))  # rows: action, columns: rewards
    draw_counts = np.zeros(2, dtype=int64)  # action0, action1 draw counts

    actions = np.zeros(trial_count, dtype=int64)
    rewards = np.zeros(trial_count)

    Q = np.array([0.5, 0.5])  # init with equal probabilities for each action

    for trial in range(trial_count):

        # compute choice probabilities using softmax
        q_soft = Q - np.max(Q)
        probabilities = np.exp(beta * q_soft) / np.sum(np.exp(beta * q_soft))

        # make choice based on choice probabilities
        actions[trial] = choose([0, 1], probabilities)

        # generate reward based on choice
        rewards[trial] = decks[actions[trial], draw_counts[actions[trial]]]
        draw_counts[actions[trial]] += 1

        # update action values
        maxQ = np.max(Q)
        delta = rewards[trial] + gamma * maxQ - Q[actions[trial]]  # prediction error
        Q[actions[trial]] += alpha * delta

    return actions, rewards


@njit
def play_worthy_bandit_states(
    alpha: float,
    beta: float,
    gamma: float,
    trial_count: int,
    deck1: np.array,
    deck2: np.array,
):
    """
    simulate state enhanced q-learner playing worthy bandit task

    parameters:
        alpha: learning rate
        beta : soft max temperature
        gamma: future discount
        trial_count : number of bandit arm pulls
        deck1: rewards for each card in the deck
        deck2: rewards for each card in the deck

    returns:
        actions (np.array.int)  : list of actions
        rewards (np.array.float): list of rewards
        stimuli (np.array.int)  : list of states
    """

    decks = np.vstack((deck1, deck2))  # rows: action, columns: trial
    draw_counts = np.zeros(2, dtype=int64)  # action0, action1 draw counts

    actions = np.zeros(trial_count, dtype=int64)
    rewards = np.zeros(trial_count)
    states = np.zeros(trial_count, dtype=int64)

    # init with equal probabilities for each action
    Q = np.ones((2, 2)) * 0.5  # 2 states, 2 actions (row, column)

    for trial in range(trial_count):

        state = get_state(draw_counts)
        states[trial] = state

        q_row = Q[state]

        # compute choice probabilities using softmax
        q_soft = q_row - np.max(q_row)  # trick for numerical stability
        probabilities = np.exp(beta * q_soft) / np.sum(np.exp(beta * q_soft))

        # make choice based on choice probabilities
        actions[trial] = choose([0, 1], probabilities)

        # generate reward based on choice
        rewards[trial] = decks[actions[trial], draw_counts[actions[trial]]]
        draw_counts[actions[trial]] += 1

        # cant update on last trial as there is no next observation
        if trial < trial_count - 1:
            # find maxQ based on next observation
            next_state = get_state(draw_counts)
            next_qrow = Q[next_state]
            maxQ = np.max(next_qrow)

            # update action values
            delta = (
                rewards[trial] + gamma * maxQ - q_row[actions[trial]]
            )  # prediction error
            Q[state, actions[trial]] += alpha * delta

    return actions, rewards, states


@njit
def get_state(draw_counts):
    GD_count, BD_count = draw_counts
    if BD_count < 30:
        state = 1  # BD best
    else:
        if BD_count > 50:
            state = 0  # GD best
        else:
            if GD_count < 20:
                state = 1
            else:
                state = 0
    return state


@njit
def play_shapetask(
    alpha: float, beta: float, gamma: float, trial_count: int, stimuli: np.array
):
    """
    play shapetask using stimuli as states
    """

    # initialize qvalues matrix and chosen actions array
    # qvalues = np.random.rand(3, 3)
    qvalues = np.repeat(1 / 3, 9).reshape(3, 3)
    actions = np.zeros(trial_count, dtype=int64)
    rewards = np.zeros(trial_count)  # shapetask doesnt have rewards but good for debug

    for trial, state in enumerate(stimuli):

        q_row = qvalues[state]

        # calculate action probabilities with softmax
        exp_qvals = np.exp(beta * q_row)
        probs = exp_qvals / np.sum(exp_qvals)

        # select action based on probabilities and save it
        action = choose([0, 1, 2], probabilities=probs)
        actions[trial] = action

        # can't get next state/observation if we're on last trial
        if trial == trial_count - 1:
            break

        if action == stimuli[trial + 1]:
            reward = 1
        else:
            reward = 0
        rewards[trial] = reward

        # find maxQ based on next observation
        next_state = stimuli[trial + 1]
        next_qrow = qvalues[next_state]
        maxQ = np.max(next_qrow)

        prediction_error = reward + gamma * maxQ - q_row[action]

        qvalues[state, action] += alpha * prediction_error

    return actions, rewards
