from dataclasses import dataclass
from typing import *


"""
An abstraction for dimensionality reduction algorithm classes
"""
@dataclass(repr=True)
class dim_reduc_function:
    name: str
    params: dict

    def __init__(self, name, f, params: dict):
        self.name: str = name
        self.f = f
        self.params: dict = params

    def apply(self, data):
        return self.f(data)

