from typing import Callable
from typing import TypeVar

from ICARUS.Core.types import FloatArray

T = TypeVar("T", float, FloatArray)


class Trajectory:
    def __init__(self, title: str, trajectory_function: Callable[..., T]) -> None:
        self.fun: Callable[..., T] = trajectory_function
        self.operating_floor: float = 10
        self.title: str = title

    def __call__(self, x: T) -> T:
        return self.fun(x)

    def first_derivative_x_fd(self, x: T) -> T:
        h = 0.0001
        return (self.fun(x + h) - self.fun(x - h)) / (2 * h)

    def second_derivative_x_fd(self, x: T) -> T:
        # Second derivative
        h = 0.0001
        return (self.fun(x + h) - 2 * self.fun(x) + self.fun(x - h)) / (h**2)

    def third_derivative_x_fd(self, x: T) -> T:
        # Third derivative
        h = 0.0001
        return (-self.fun(x - 2 * h) + 2 * self.fun(x - h) - 2 * self.fun(x + h) + self.fun(x + 2 * h)) / (2 * h**3)
