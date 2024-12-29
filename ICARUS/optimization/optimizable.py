from abc import ABC
from abc import abstractmethod
from typing import Any


class Optimizable(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_property(self, name: str) -> Any:
        pass

    @abstractmethod
    def set_property(self, name: str, value: Any) -> None:
        pass
