from functools import partial

import jax.numpy as jnp
from jax import jit
from jax import lax

from ..first_order_system import DynamicalSystem
from .base_integrator import Integrator


class RK45Integrator(Integrator):
    """
    This class implements the Runge-Kutta Fehlberg 4(5) method for solving ODEs
    RK45 is a 5th order method with an embedded 4th order method for error estimation
    """

    def __init__(self, dt: float, system: DynamicalSystem) -> None:
        self.name = "RK45"
        self.type = "Explicit"
        self.dt = dt
        self.system = system

    @partial(jit, static_argnums=(0,))
    def step(self, t: float, x: jnp.ndarray, dt: float) -> tuple[jnp.ndarray, float]:
        # Implement the Runge-Kutta Fehlberg 4(5) method
        tol = 1e-6

        k1 = self.system(t, x)
        k2 = self.system(t + (1 / 4) * dt, x + (1 / 4) * dt * k1)
        k3 = self.system(t + (3 / 8) * dt, x + (3 / 32) * dt * k1 + (9 / 32) * dt * k2)
        k4 = self.system(
            t + (12 / 13) * dt,
            x + (1932 / 2197) * dt * k1 - (7200 / 2197) * dt * k2 + (7296 / 2197) * dt * k3,
        )
        k5 = self.system(
            t + dt,
            x + (439 / 216) * dt * k1 - 8 * dt * k2 + (3680 / 513) * dt * k3 - (845 / 4104) * dt * k4,
        )
        k6 = self.system(
            t + (1 / 2) * dt,
            x
            - (8 / 27) * dt * k1
            + 2 * dt * k2
            - (3544 / 2565) * dt * k3
            + (1859 / 4104) * dt * k4
            - (11 / 40) * dt * k5,
        )

        # Compute the 4th and 5th order estimates
        x_4 = x + (25 / 216) * dt * k1 + (1408 / 2565) * dt * k3 + (2197 / 4104) * dt * k4 - (1 / 5) * dt * k5
        x_5 = (
            x
            + (16 / 135) * dt * k1
            + (6656 / 12825) * dt * k3
            + (28561 / 56430) * dt * k4
            - (9 / 50) * dt * k5
            + (2 / 55) * dt * k6
        )

        # Compute the error
        error = jnp.abs(x_5 - x_4)

        # Adjust step size based on error
        delta = 0.84 * (tol / jnp.max(error)) ** 0.25
        delta = jnp.clip(delta, 0.1, 1.1)
        dt = dt * delta  # type: ignore
        return x_4, dt

    def simulate(self, x0: jnp.ndarray, t0: float, tf: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        times, trajectory = self._simulate(x0, t0, tf)
        return times, trajectory

    def _simulate(self, x0: jnp.ndarray, t0: float, tf: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Create a loop using lax.while that integrates the system using the RK45 method and
        # stores the results in the trajectory array

        # Preallocate the trajectory array
        # Since we don't know the number of steps, we will use a python list to store the trajectory
        trajectory = [x0]
        times = [t0]
        x = x0
        dt = self.dt
        t = t0
        itera = 0

        while t < tf:
            x, dt = self.step(t, x, dt)
            t += dt
            trajectory.append(x)
            times.append(t)
            itera += 1

        return jnp.array(times), jnp.array(trajectory)
