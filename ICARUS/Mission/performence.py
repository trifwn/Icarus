from typing import Any
from typing import Callable


class Fitness:
    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func

    def getFitness(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)
