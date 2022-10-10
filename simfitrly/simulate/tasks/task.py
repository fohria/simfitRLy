"""
    this is the baseclass for tasks. extend it to create new task.
"""
from dataclasses import dataclass, astuple, asdict


@dataclass
class Task:
    def get_tuple(self):
        return astuple(self)

    def get_dict(self):
        return asdict(self)

    def to_filename(self):
        params = self.get_dict()
        taskname = params.pop("name")
        filename = taskname
        for param, value in params.items():
            filename += "-" + param + "-" + str(value)
        return filename
