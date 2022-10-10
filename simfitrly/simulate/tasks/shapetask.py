"""
    shape sequence task.
    in this task, participant will see a shape on the screen and they will have three action options. the goal is to predict what the next shape will be.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from .task import Task


@dataclass
class Shapetask(Task):
    """
    name            : name of the task/experiment
    version         :
        bob-nr      : bag of bags no repeat
        bob         : bag of bags
        random      : random bags
        random-fixed: random bags, fixed sequence
    trial_count     : number of experimental trials
    bagsize         : size of one "bag", i.e.
    """

    # TODO check nbr trials and bagsize adds up
    # for example trial_count = 5 only works with bagsize=1 or bagsize=5

    version: str
    trial_count: int
    bagsize: int = 3
    name: str = "shapetask"

    def __post_init__(self):
        self.stimuli = generate_stimuli_sequence(self, stimtype="integer")
        task_versions = ["bob-nr", "bob", "random", "random-fixed"]
        if self.version not in task_versions:
            raise ValueError(
                f"wrong task version name! should be one of {task_versions=}"
            )

    def get_stimuli(self):
        return self.stimuli

    def generate_stimuli(self, stimtype="integer"):
        # if we want to update stimuli for whatever reason
        return generate_stimuli_sequence(self, stimtype)

    def score_experiment(self, dataframe):
        return score_experiment(dataframe, self)


def generate_stimuli_sequence(params, stimtype="integer"):
    """
    generate a stimuli sequence for the shape sequence task

    stimtype can be `integer` or `string`, where former gives 0,1,2 and latter
    gives circle, triangle, square
    """

    taskversion = params.version

    sequence = []

    if stimtype == "string":
        shapes = {0: "circle", 1: "triangle", 2: "square"}
    if stimtype == "integer":
        shapes = {0: 0, 1: 1, 2: 2}

    trial_count = params.trial_count
    bagsize = params.bagsize
    nbr_of_bags = int(trial_count / bagsize)

    # bag of bags version; shape can repeat maximum 6 times in a row
    if taskversion == "bob":

        nbr_of_bobs = int(trial_count / (bagsize * len(shapes)))
        bag_o_bag = shapes

        for b0b in range(nbr_of_bobs):
            current_bob = list(bag_o_bag.keys())

            while len(current_bob) > 0:
                bag = np.random.choice(current_bob)

                for trial in range(bagsize):
                    sequence.append(shapes[bag])

                current_bob.pop(current_bob.index(bag))

    # random version; shape repeats minimum 3 times and no theoretical max
    if taskversion == "random":

        for _ in range(nbr_of_bags):
            bag = np.random.choice(list(shapes.keys()))

            for trial in range(bagsize):
                sequence.append(shapes[bag])

    # bag of bags no repeat; shape can repeat maximum 3 times
    if taskversion == "bob-nr":

        bag_o_bag = shapes

        nbr_of_bobs = int(trial_count / (bagsize * len(shapes)))

        last_bag = -1
        for b0b in range(nbr_of_bobs):
            current_bob = list(bag_o_bag.keys())

            while len(current_bob) > 0:
                bag = np.random.choice(current_bob)

                while bag == last_bag:
                    bag = np.random.choice(current_bob)

                for trial in range(bagsize):
                    sequence.append(shapes[bag])

                last_bag = current_bob.pop(current_bob.index(bag))

    # random fixed sequence; used in pilot experiment
    if taskversion == "random-fixed":
        sequence = np.loadtxt(
            "simfitrly/simulate/tasks/static_data/random-fixed.csv",
            dtype=int,
            converters={0: colormap},
            usecols=0,
            delimiter=",",
            encoding="utf-8",
        )
        sequence = [shapes[x] for x in sequence]

    return np.array(sequence)


def score_experiment(tidy_df, taskparams):
    """
    calculate relevant scores for this experimental task.
    df should be a tidy dataframe.

    CAUTION: this function is currently only useable for a single subject.

    """

    df = tidy_df.copy()

    # get subject count if we are sent many of them
    subject_count = int(len(df) / taskparams.trial_count)

    # check if action correctly predicted next stimulus
    # shift values of action column down one cell to compare to next stimulus
    compare_to_next = df["stimulus"] == df["action"].shift(1)
    # shift back up to record if action on current trial is correct prediction
    df["correct"] = compare_to_next.shift(-1)
    # for later plotting etc it's useful to have float
    df["correct"] = df["correct"].astype("float")
    # in case we have many subjects correct cant be set for last trial
    df.loc[df.trial == taskparams.trial_count - 1, "correct"] = np.nan

    # does participant predict next shape to differ from the current stimulus?
    df["shift_predict"] = df["action"] != df["stimulus"]
    df["shift_predict"] = df["shift_predict"].astype("float")

    # if prediction on this trial is same as prediction on last trial and
    # last trial was correct => WIN STAY
    last_same_this_trial = df["action"] == df["action"].shift(1)
    last_correct = df["correct"].shift(1)
    # python treats true as 1 and false as 0
    winstay = last_same_this_trial + last_correct
    # winstay is thus where both are true
    df["winstay"] = winstay == 2
    # winstay doesn't make sense for the first trial so set to nan
    # winstay_index = df.columns.get_loc('winstay')
    # df.iloc[0, winstay_index] = np.nan
    df.loc[df.trial == 0, "winstay"] = np.nan
    # np.nan casts column as float so convert to int
    # actually doesnt it seems so have to convert to float for later groupbys
    df.winstay = pd.to_numeric(df.winstay)
    # pd.NA doesnt work as that will turn into object series
    # not sure this is strictly needed, could let it be float
    # df['winstay'] = df['winstay'].astype('Int64')
    # assert below doesnt work with <NA> apparently so fuck it

    # if prediction on this trial is different from prediction on last trial and
    # last trial was incorrect = LOSE SHIFT
    last_diff_this_trial = df["action"] != df["action"].shift(1)
    last_incorrect = [0 if x else 1 for x in df["correct"].shift(1)]
    lose_shift = last_diff_this_trial + last_incorrect
    df["loseshift"] = lose_shift == 2
    # loseshift_index = df.columns.get_loc('loseshift')
    # df.iloc[0, loseshift_index] = np.nan
    df.loc[df.trial == 0, "loseshift"] = np.nan
    # df['loseshift'] = df['loseshift'].astype('Int64')
    # convert to float for later groupby possibility
    df.loseshift = pd.to_numeric(df.loseshift)

    # final sanity check that no row is both winstay and loseshift
    check = (df["winstay"] + df["loseshift"]) > 1
    assert sum(check) == 0, "one or more trials is both winstay and loseshift"

    # for later analysis and plotting, add shape positions
    # shape position is the the order of presentation within a bag
    bag_count = int(taskparams.trial_count / taskparams.bagsize)
    shape_position = np.tile(np.arange(1, taskparams.bagsize + 1), bag_count)
    df["shape_position"] = np.tile(shape_position, subject_count)

    # add bag of bags (bob) number/occurence
    bob_count = int(bag_count / 3)  # always have 3 shapes currently
    bob_number = np.repeat(
        np.arange(1, bob_count + 1), taskparams.trial_count / bob_count
    )
    df["bob_number"] = np.tile(bob_number, subject_count)

    return df


def colormap(colorname):
    if colorname == "blue":
        return 0
    if colorname == "red":
        return 1
    if colorname == "green":
        return 2
