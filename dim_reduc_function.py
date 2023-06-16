from abc import ABC


class dim_reduc_function(ABC):

    def __init__(self, name, f, params: dict):
        self.name: str = name
        self.f = f
        self.params: dict = params

    def apply(self, data):
        return self.f(data)

