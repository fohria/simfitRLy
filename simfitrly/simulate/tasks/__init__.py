"""
    hello! i'm needed to import classes and functions so we can write:
    simulate.tasks.Bandit()
    instead of
    simulate.tasks.bandit.Bandit()
    when we do a module level import like:
    import simulate
    in the root script
"""

from .bandit import Bandit
from .reversal_bandit import ReversalBandit
from .worthy_bandit import WorthyBandit
from .shapetask import Shapetask
