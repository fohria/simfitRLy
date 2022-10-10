from dataclasses import dataclass, astuple, asdict, field

import numpy as np  # only need this because seql2
from scipy.optimize import minimize  # only need this because seql2

# bandit likelihoods
from .model_store.ql2.ql2_bandit import likelihood as ql2_bandit
from .model_store.random_bias.random_bias_bandit import likelihood as random_bias_bandit
from .model_store.seql2.seql2_reversal_bandit import likelihood as seql2_revbandit
from .model_store.ql_dual_alpha.ql_dual_alpha_bandit import (
    likelihood as ql_dual_alpha_bandit,
)
from .model_store.ql_dual_update.ql_dual_update_bandit import (
    likelihood as ql_dual_update_bandit,
)
from .model_store.hmm.hmm_bandit import likelihood as hmm_bandit
from .model_store.hmm.hmm_delta_bandit import likelihood as hmm_delta_bandit
from .model_store.hmm.hmm_worthy_bandit import likelihood as hmm_worthy_bandit
from .model_store.hmm.hmm_delta_worthy_bandit import (
    likelihood as hmm_delta_worthy_bandit,
)

# shapetask likelihoods
from .model_store.hrl.hrl_shapetask import likelihood as hrl_shapetask
from .model_store.ql3.ql3_shapetask import likelihood as ql3_shapetask
from .model_store.seql3.seql3_shapetask import likelihood as seql3_shapetask
from .model_store.srtd.srtd_shapetask import likelihood as srtd_shapetask
from .model_store.random_bias.random_bias_shapetask import (
    likelihood as random_bias_shapetask,
)

# fit methods
from . import mle
from . import vb
from . import mcmc

# TODO! parse/validate method name and model name
# TODO! we may be able to put `fit_bandit` function in base class
# TODO! if we put in base class we can rely on checking model dicts if model/task combo has been implemented or not


@dataclass
class Model:
    """
    Behavioural model base class. A model is an algorithm that describes learning behaviour.
    """

    # method    : name of the method (mle/mcmc/mcmc-hierarchical)
    # model     : what specific model to use
    # model_file: path and filename to modelfile (populated when fitresults returns)
    # return_full_fitdata: also return stan fit objects, not just summary. default is False

    # method             : str
    # model              : str
    # model_file         : str  = ''
    # return_full_fitdata: bool = False

    def get_tuple(self):
        return astuple(self)

    def get_dict(self):
        return asdict(self)


@dataclass
class QL2(Model):
    """
    fit QL2 model to data from a task

    use the input parameters to adjust how fitting will be done
    """

    name: str = "QL2"
    parameters: tuple = ("alpha", "beta")
    stan_models: dict = field(default_factory=dict)
    mle_models: dict = field(default_factory=dict)

    def __post_init__(self):
        self.stan_models = {
            "bandit": {
                "bayes_ind": "simfitrly/fit/model_store/ql2/ql2_bandit_new.stan",
                "bayes_hier": "simfitrly/fit/model_store/ql2/ql2_bandit_hier.stan",
            }
        }
        self.mle_models = {
            "bandit": {
                "model_file": "simfitrly/fit/model_store/ql2/ql2_bandit.py",
                "likelihood": ql2_bandit,
                "start_guess": [(0, 1), (0, 20)],  # alpha, beta
                "bounds": [(0, 1), (0, 50)],  # alpha, beta
            }
        }

    def fit_bandit_mle(self, actions, rewards):
        return mle.fit_single_subject_bandit(
            self.mle_models["bandit"]["likelihood"],
            self.mle_models["bandit"]["start_guess"],
            self.mle_models["bandit"]["bounds"],
            actions,
            rewards,
        )

    def fit_bandit_vb(self, actions, rewards, subject_id):
        return vb.fit_single_subject_bandit(
            self.stan_models["bandit"]["bayes_ind"], actions, rewards, subject_id
        )

    def fit_bandit_mcmc(self, actions, rewards, subject_id, parchains=1):
        return mcmc.fit_single_subject_bandit(
            self.stan_models["bandit"]["bayes_ind"],
            actions,
            rewards,
            subject_id,
            parallel_chains=parchains,
        )


@dataclass
class RandomBias(Model):
    """
    fit random bias model to data from a task

    use the input parameters to adjust how fitting will be done
    """

    name: str = "RandomBias"
    parameters: tuple = ("bias",)  # comma needed or it will become string
    stan_models: dict = field(default_factory=dict)
    mle_models: dict = field(default_factory=dict)

    def __post_init__(self):
        self.stan_models = {
            "bandit": {
                "bayes_ind": "simfitrly/fit/model_store/random_bias/random_bias_bandit.stan",
                "bayes_hier": "simfitrly/fit/model_store/random_bias/random_bias_bandit_hier.stan",
            }
        }
        self.mle_models = {
            "bandit": {
                "model_file": "simfitrly/fit/model_store/random_bias/random_bias_bandit.py",
                "likelihood": random_bias_bandit,
                "start_guess": [(0, 1)],  # bias
                "bounds": [(0, 1)],  # bias
            },
            "shapetask": {
                "model_file": "simfitrly/fit/model_store/random_bias/random_bias_shapetask.py",
                "likelihood": random_bias_shapetask,
                "start_guess": [(0, 1), (0, 1)],
                "bounds": [(0, 1), (0, 1)],
            },
        }

    # TODO: so below three are exactly same now as in ql2 class and should work?
    def fit_bandit_mle(self, actions, rewards):
        return mle.fit_single_subject_bandit(
            self.mle_models["bandit"]["likelihood"],
            self.mle_models["bandit"]["start_guess"],
            self.mle_models["bandit"]["bounds"],
            actions,
            rewards,
        )

    def fit_bandit_vb(self, actions, rewards, subject_id):
        return vb.fit_single_subject_bandit(
            self.stan_models["bandit"]["bayes_ind"], actions, rewards, subject_id
        )

    def fit_bandit_mcmc(self, actions, rewards, subject_id, parchains=1):
        return mcmc.fit_single_subject_bandit(
            self.stan_models["bandit"]["bayes_ind"],
            actions,
            rewards,
            subject_id,
            parallel_chains=parchains,
        )

    def fit_shapetask_mle(self, actions, rewards, stimuli):
        return mle.fit_single_subject_shapetask(
            self.mle_models["shapetask"]["likelihood"],
            self.mle_models["shapetask"]["start_guess"],
            self.mle_models["shapetask"]["bounds"],
            actions,
            rewards,
            stimuli,
        )


@dataclass
class SEQL2(Model):
    """
    fit seql2 model to data from a task

    use the input parameters to adjust how fitting will be done
    """

    name: str = "SEQL2"
    parameters: tuple = ("alpha", "beta")
    mle_models: dict = field(default_factory=dict)

    def __post_init__(self):
        self.mle_models = {
            "reversal_bandit": {
                "model_file": "fit/model_store/seql2/seql2_reversal_bandit.py",
                "likelihood": seql2_revbandit,  # TODO fix this
                "start_guess": [(0, 1), (0, 20)],  # alpha, beta
                "bounds": [(0, 1), (0, 50)],  # alpha, beta
            }
        }

    def fit_reversal_bandit_mle(self, actions, rewards, stimuli):

        guesses_per_loop = 10

        starting_guess_bounds = self.mle_models["reversal_bandit"]["start_guess"]
        fit_bounds = self.mle_models["reversal_bandit"]["bounds"]
        model_likelihood = self.mle_models["reversal_bandit"]["likelihood"]

        best_loglike = 999999  # best log-likelihood will be much lower
        best_result = None
        rng = np.random.RandomState()

        for _ in range(guesses_per_loop):

            start_guess = [
                rng.uniform(lower, upper) for lower, upper in starting_guess_bounds
            ]

            fitresult = minimize(
                model_likelihood,
                start_guess,
                args=(actions, rewards, stimuli),
                bounds=fit_bounds,
            )

            if fitresult.fun < best_loglike and fitresult.success:
                best_result = fitresult
                best_loglike = fitresult.fun

        return best_result


@dataclass
class QLDualAlpha(Model):
    """
    fit QL2 dual alpha model to data from a task

    use the input parameters to adjust how fitting will be done
    TODO: what input parameters?! haha
    """

    name: str = "QLDualAlpha"
    parameters: tuple = ("alpha_pos", "alpha_neg", "beta")
    mle_models: dict = field(default_factory=dict)

    def __post_init__(self):
        self.mle_models = {
            "bandit": {
                "model_file": "fit/model_store/ql_dual_alpha/ql_dual_alpha_bandit.py",
                "likelihood": ql_dual_alpha_bandit,
                "start_guess": [(0, 1), (0, 1), (0, 20)],  # alphapos, alphaneg, beta
                "bounds": [(0, 1), (0, 1), (0, 50)],  # alphapos, alphaneg, beta
            }
        }

    def fit_bandit_mle(self, actions, rewards):
        return mle.fit_single_subject_bandit(
            self.mle_models["bandit"]["likelihood"],
            self.mle_models["bandit"]["start_guess"],
            self.mle_models["bandit"]["bounds"],
            actions,
            rewards,
        )


@dataclass
class QLDualUpdate(Model):
    """
    fit QL2 dual update model to data from a task
    """

    name: str = "QLDualUpdate"
    parameters: tuple = ("alpha", "beta")
    mle_models: dict = field(default_factory=dict)

    def __post_init__(self):
        self.mle_models = {
            "bandit": {
                "model_file": "fit/model_store/ql_dual_update/ql_dual_update_bandit.py",
                "likelihood": ql_dual_update_bandit,
                "start_guess": [(0, 1), (0, 20)],  # alpha, beta
                "bounds": [(0, 1), (0, 50)],  # alpha, beta
            }
        }

    def fit_bandit_mle(self, actions, rewards):
        return mle.fit_single_subject_bandit(
            self.mle_models["bandit"]["likelihood"],
            self.mle_models["bandit"]["start_guess"],
            self.mle_models["bandit"]["bounds"],
            actions,
            rewards,
        )


@dataclass
class HMM(Model):
    """
    fit HMM model to data from a task
    """

    name: str = "HMM"
    parameters: tuple = ("gamma",)  # comma needed or it will become string
    mle_models: dict = field(default_factory=dict)

    def __post_init__(self):
        self.mle_models = {
            "bandit": {
                "model_file": "fit/model_store/hmm/hmm_bandit.py",
                "likelihood": hmm_bandit,
                "start_guess": [(0, 1)],  # gamma
                "bounds": [(0, 1)],  # gamma
            }
        }

    def fit_bandit_mle(self, actions, rewards):
        return mle.fit_single_subject_bandit(
            self.mle_models["bandit"]["likelihood"],
            self.mle_models["bandit"]["start_guess"],
            self.mle_models["bandit"]["bounds"],
            actions,
            rewards,
        )


@dataclass
class HMMDelta(Model):
    """
    fit HMM model to data from a task
    """

    name: str = "HMMDelta"
    parameters: tuple = ("gamma", "delta")
    mle_models: dict = field(default_factory=dict)

    def __post_init__(self):
        self.mle_models = {
            "bandit": {
                "model_file": "fit/model_store/hmm/hmm_delta_bandit.py",
                "likelihood": hmm_delta_bandit,
                "start_guess": [(0, 1), (0, 1)],  # gamma, delta
                "bounds": [(0, 1), (0, 1)],  # gamma, delta
            }
        }

    def fit_bandit_mle(self, actions, rewards):
        return mle.fit_single_subject_bandit(
            self.mle_models["bandit"]["likelihood"],
            self.mle_models["bandit"]["start_guess"],
            self.mle_models["bandit"]["bounds"],
            actions,
            rewards,
        )


@dataclass
class HMMWorthy(Model):
    """
    fit HMM model to data from a task
    """

    name: str = "HMM"
    parameters: tuple = ("gamma",)  # comma needed or it will become string
    mle_models: dict = field(default_factory=dict)

    def __post_init__(self):
        self.mle_models = {
            "bandit": {
                "model_file": "fit/model_store/hmm/hmm_worthy_bandit.py",
                "likelihood": hmm_worthy_bandit,
                "start_guess": [(0, 1)],  # gamma
                "bounds": [(0, 1)],  # gamma
            }
        }

    def fit_bandit_mle(self, actions, rewards):
        return mle.fit_single_subject_bandit(
            self.mle_models["bandit"]["likelihood"],
            self.mle_models["bandit"]["start_guess"],
            self.mle_models["bandit"]["bounds"],
            actions,
            rewards,
        )


@dataclass
class HMMDeltaWorthy(Model):
    """
    fit HMM model to data from a task
    """

    name: str = "HMMDelta"
    parameters: tuple = ("gamma", "delta")
    mle_models: dict = field(default_factory=dict)

    def __post_init__(self):
        self.mle_models = {
            "bandit": {
                "model_file": "fit/model_store/hmm/hmm_delta_worthy_bandit.py",
                "likelihood": hmm_delta_worthy_bandit,
                "start_guess": [(0, 1), (0, 1)],  # gamma, delta
                "bounds": [(0, 1), (0, 1)],  # gamma, delta
            }
        }

    def fit_bandit_mle(self, actions, rewards):
        return mle.fit_single_subject_bandit(
            self.mle_models["bandit"]["likelihood"],
            self.mle_models["bandit"]["start_guess"],
            self.mle_models["bandit"]["bounds"],
            actions,
            rewards,
        )


@dataclass
class HRL(Model):
    """
    fit HRL model based on Eckstein & Collins 2020 to data from shapetask
    """

    name: str = "HRL"
    parameters: tuple = ("alpha_low", "alpha_high", "beta_low", "beta_high")
    mle_models: dict = field(default_factory=dict)

    def __post_init__(self):
        self.mle_models = {
            "shapetask": {
                "model_file": "fit/model_store/hrl/hrl_shapetask.py",
                "likelihood": hrl_shapetask,
                "start_guess": [(0, 1), (0, 1), (0, 20), (0, 20)],
                "bounds": [(0, 1), (0, 1), (0, 50), (0, 50)],
            }
        }

    def fit_shapetask_mle(self, actions, rewards, stimuli):
        return mle.fit_single_subject_shapetask_HRL(
            self.mle_models["shapetask"]["likelihood"],
            self.mle_models["shapetask"]["start_guess"],
            self.mle_models["shapetask"]["bounds"],
            actions,
            rewards,
            stimuli,
        )


@dataclass
class QL3(Model):
    """
    fit Q-learning with 3 parameters to data from shapetask
    """

    name: str = "QL3"
    parameters: tuple = ("alpha", "beta", "gamma")
    mle_models: dict = field(default_factory=dict)

    def __post_init__(self):
        self.mle_models = {
            "shapetask": {
                "model_file": "fit/model_store/ql3/ql3_shapetask.py",
                "likelihood": ql3_shapetask,
                "start_guess": [(0, 1), (0, 20), (0, 1)],
                "bounds": [(0, 1), (0, 50), (0, 1)],
            }
        }

    def fit_shapetask_mle(self, actions, rewards, stimuli):
        return mle.fit_single_subject_shapetask(
            self.mle_models["shapetask"]["likelihood"],
            self.mle_models["shapetask"]["start_guess"],
            self.mle_models["shapetask"]["bounds"],
            actions,
            rewards,
            stimuli,
        )


@dataclass
class SEQL3(Model):
    """
    fit State Enhanced Q-learning with 3 parameters to data from shapetask
    """

    name: str = "SEQL3"
    parameters: tuple = ("alpha", "beta", "gamma")
    mle_models: dict = field(default_factory=dict)

    def __post_init__(self):
        self.mle_models = {
            "shapetask": {
                "model_file": "fit/model_store/seql3/seql3_shapetask.py",
                "likelihood": seql3_shapetask,
                "start_guess": [(0, 1), (0, 20), (0, 1)],
                "bounds": [(0, 1), (0, 50), (0, 1)],
            }
        }

    def fit_shapetask_mle(self, actions, rewards, stimuli):
        return mle.fit_single_subject_shapetask(
            self.mle_models["shapetask"]["likelihood"],
            self.mle_models["shapetask"]["start_guess"],
            self.mle_models["shapetask"]["bounds"],
            actions,
            rewards,
            stimuli,
        )


@dataclass
class SRTD(Model):
    """
    fit SRTD as per Russek et al 2017 to data from shapetask
    """

    name: str = "SRTD"
    parameters: tuple = ("alpha_sr", "alpha_w", "beta", "gamma")
    mle_models: dict = field(default_factory=dict)

    def __post_init__(self):
        self.mle_models = {
            "shapetask": {
                "model_file": "fit/model_store/srtd/srtd_shapetask.py",
                "likelihood": srtd_shapetask,
                "start_guess": [(0, 1), (0, 1), (0, 20), (0, 1)],
                "bounds": [(0, 1), (0, 1), (0, 50), (0, 1)],
            }
        }

    def fit_shapetask_mle(self, actions, rewards, stimuli):
        bagsize = 3  # hard coded for now
        maze = np.arange(9).reshape(3, 3).T
        state_sequence = [maze[bag] for bag in stimuli[::bagsize]]
        state_sequence = np.ravel(state_sequence)
        return mle.fit_single_subject_shapetask(
            self.mle_models["shapetask"]["likelihood"],
            self.mle_models["shapetask"]["start_guess"],
            self.mle_models["shapetask"]["bounds"],
            actions,
            rewards,
            state_sequence,
        )
