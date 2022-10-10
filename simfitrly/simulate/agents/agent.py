from dataclasses import dataclass, astuple, asdict

"""
    this is the base (data)class for agents. extend it to create a new agent.
"""


@dataclass
class Agent:
    def get_tuple(self):
        return astuple(self)

    def get_dict(self):
        return asdict(self)
