"""
    hello! i'm needed to import classes and functions so we can write:
    simulate.agents.QL()
    instead of
    simulate.agents.ql.QL()
    when we do a module level import like:
    import simulate
    in the root script
"""

from .ql import QL
from .random_bias import RandomBias
from .ql_dual_alpha import QLDualAlpha
from .ql_dual_update import QLDualUpdate
from .hmm import HMM
from .hmm_delta import HMMDelta
from .seql import SEQL
from .hrl import HRL
from .srtd import SRTD
