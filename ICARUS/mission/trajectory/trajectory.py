from typing import Callable

from ICARUS.core.types import AnyFloat
from ICARUS.core.types import FloatArray


class Trajectory:
    def __init__(self, title: str, trajectory_function: Callable[[AnyFloat], FloatArray]) -> None:
        self.fun: Callable[[AnyFloat], FloatArray] = trajectory_function
        self.operating_floor: float = 10.0
        self.title: str = title

    def __call__(self, x: float | FloatArray) -> FloatArray:
        return self.fun(x)

    def first_derivative_x_fd(self, x: float | FloatArray) -> FloatArray:
        h = 0.0001
        return (self.fun(x + h) - self.fun(x - h)) / (2 * h)

    def second_derivative_x_fd(self, x: float | FloatArray) -> FloatArray:
        # Second derivative
        h = 0.0001
        return (self.fun(x + h) - 2 * self.fun(x) + self.fun(x - h)) / (h**2)

    def third_derivative_x_fd(self, x: float | FloatArray) -> FloatArray:
        # Third derivative
        h = 0.0001
        res: FloatArray = (-self.fun(x - 2 * h) + 2 * self.fun(x - h) - 2 * self.fun(x + h) + self.fun(x + 2 * h)) / (
            2 * h**3
        )
        return res
