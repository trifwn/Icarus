from typing import Any


class OptimizationCallback:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def setup(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError
