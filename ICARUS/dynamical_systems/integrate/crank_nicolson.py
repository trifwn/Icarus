from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import lax

from ..base_system import DynamicalSystem
from .base_integrator import Integrator


class CrankNicolsonIntegrator(Integrator):
    def __init__(self, dt: float, system: DynamicalSystem, tol: float = 1e-6) -> None:
        self.name = "Crank-Nicolson"
        self.type = "Semi-Implicit"
        self.dt = dt
        self.system = system
        self.tol = tol

    @partial(jit, static_argnums=(0,))
    def step(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        def error(x1: jnp.ndarray) -> jnp.ndarray:
            """Calculates the error for the current state estimate."""
            k1 = self.system(t, x)
            k2 = self.system(t + self.dt, x1)
            predicted_state = x + 0.5 * self.dt * (k1 + k2)
            return predicted_state - x - self.dt * self.system(t + 0.5 * self.dt, predicted_state)

        def cond_fun(er: jnp.ndarray) -> jnp.ndarray:
            """Loop condition based on error norm."""
            return jnp.linalg.norm(er) > self.tol

        def body_fun(x1: jnp.ndarray) -> jnp.ndarray:
            """Iteratively solve for the full step solution."""
            j = self.system.jacobian(t + 0.5 * self.dt, x1)
            er = error(x1)
            return x1 - jnp.linalg.solve(j, er)

        x_next = lax.while_loop(
            cond_fun,
            body_fun,
            x + 0.5 * self.dt * self.system(t, x),
        )

        return x_next

    def simulate(
        self,
        x0: jnp.ndarray,
        t0: float,
        tf: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        num_steps = jnp.ceil((tf - t0) / self.dt).astype(int)

        trajectory = jnp.zeros((num_steps + 1, x0.shape[0]))
        trajectory = trajectory.at[0].set(x0)
        times = np.linspace(start=t0, stop=tf, num=num_steps + 1)
        times, trajectory = self._simulate(trajectory, times, num_steps)
        return times, trajectory

    @partial(jit, static_argnums=(0,))
    def _simulate(
        self,
        trajectory: jnp.ndarray,
        times: jnp.ndarray,
        num_steps: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Create a loop using lax.fori_loop that integrates the system using the Crank-Nicolson method and
        # stores the results in the trajectory array
        def body(
            i: int,
            args: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            times, trajectory = args
            t = times[i - 1] + self.dt
            x = trajectory[i - 1]
            trajectory = trajectory.at[i].set(self.step(t, x))
            return times, trajectory

        times, trajectory = lax.fori_loop(1, num_steps + 1, body, (times, trajectory))
        return times, trajectory
