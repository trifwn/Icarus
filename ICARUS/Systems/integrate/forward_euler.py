from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import lax

from ..first_order_system import DynamicalSystem
from .base_integrator import Integrator


class ForwardEulerIntegrator(Integrator):
    def __init__(self, dt: float, system: DynamicalSystem) -> None:
        self.name = "Forward Euler"
        self.type = "Explicit"
        self.dt = dt
        self.system = system

    @partial(jit, static_argnums=(0,))
    def step(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return x + self.dt * self.system(t, x)

    def simulate(self, x0: jnp.ndarray, t0: float, tf: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        # If the number of steps is not an integer, we will use the ceil function to round up to the nearest integer
        num_steps = jnp.ceil((tf - t0) / self.dt).squeeze().astype(int)

        # Preallocate the trajectory array
        trajectory = jnp.zeros((num_steps + 1, x0.shape[0]))
        trajectory = trajectory.at[0].set(x0)

        times = jnp.linspace(start=t0, stop=tf, num=num_steps + 1)  # type: ignore
        times, trajectory = self._simulate(trajectory, times, num_steps)

        return times, trajectory

    @partial(jit, static_argnums=(0,))
    def _simulate(self, trajectory: jnp.ndarray, times: jnp.ndarray, num_steps: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Create a loop using lax.fori_loop that integrates the system using the forward Euler method and
        # stores the results in the trajectory array
        def body(i, args):
            times, trajectory = args
            x = trajectory[i - 1]
            t = times[i - 1]
            step = self.step(t + i * self.dt, x)
            trajectory = trajectory.at[i].set(step)
            return times, trajectory

        times, trajectory = lax.fori_loop(1, num_steps + 1, body, (times, trajectory))
        return times, trajectory
