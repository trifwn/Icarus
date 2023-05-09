from typing import Callable


class Fitness:
    def __init__(self, func: Callable) -> None:
        self.func = func

    def getFitness(self, *args, **kwargs):
        return self.func(*args, **kwargs)
