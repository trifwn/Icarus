from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import lax

from ..base_system import DynamicalSystem
from .base_integrator import Integrator


class RK4Integrator(Integrator):
    def __init__(self, dt: float, system: DynamicalSystem) -> None:
        self.name = "RK4"
        self.type = "Explicit"
        self.dt = dt
        self.system = system

    @partial(jit, static_argnums=(0,))
    def step(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        k1 = self.system(t, x)
        k2 = self.system(t + self.dt / 2, x + (self.dt / 2) * k1)
        k3 = self.system(t + self.dt / 2, x + (self.dt / 2) * k2)
        k4 = self.system(t + self.dt, x + self.dt * k3)

        return x + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def simulate(self, x0: jnp.ndarray, t0: float, tf: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        num_steps = jnp.ceil((tf - t0) / self.dt).astype(int)
        x = x0
        t = t0
        trajectory = jnp.zeros((num_steps + 1, x0.shape[0]))
        trajectory = trajectory.at[0].set(x0)

        times = np.linspace(start=t0, stop=tf, num=num_steps + 1)
        times, trajectory = self._simulate(trajectory, times, num_steps)
        return times, trajectory

    @partial(jit, static_argnums=(0,))
    def _simulate(self, trajectory: jnp.ndarray, times: jnp.ndarray, num_steps: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Create a loop using lax.fori_loop that integrates the system using the RK4 method and
        # stores the results in the trajectory array
        def body(i: int, args: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
            times, trajectory = args
            x = trajectory[i - 1]
            t = times[i - 1]
            trajectory = trajectory.at[i].set(self.step(t + i * self.dt, x))
            return times, trajectory

        times, trajectory = lax.fori_loop(1, num_steps + 1, body, (times, trajectory))
        return times, trajectory
