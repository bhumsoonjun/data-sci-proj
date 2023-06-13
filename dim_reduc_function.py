from typing import *
from abc import ABC, abstractmethod
from dataclasses import dataclass

class dim_reduc_function(ABC):

    def __init__(self, name, f, params: dict, instance_num: int):
        self.instance_num = instance_num
        self.name: str = name
        self.f = f
        self.params: dict = params

    def apply(self, data):
        return self.f(data)

