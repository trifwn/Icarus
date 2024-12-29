from abc import ABC
from abc import abstractmethod
from typing import Any


class OptimizationCallback(ABC):
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def setup(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError
