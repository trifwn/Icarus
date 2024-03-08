from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import lax

from ..base_system import DynamicalSystem
from .base_integrator import Integrator


class BackwardEulerIntegrator(Integrator):
    def __init__(
        self,
        dt: float,
        system: DynamicalSystem,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        self.name = "Backward Euler"
        self.type = "Implicit"
        self.dt: float = dt
        self.system = system
        # self.system -> f(t,x)
        # self.system.jacobian -> jacobian: df(t,x)/dx
        self.max_iter = max_iter
        self.tol = tol

    @partial(jit, static_argnums=(0,))
    def step(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        # Initial guess for the solution
        x_new = x + self.dt * self.system(t, x)

        # Define the loop body
        def body(args):
            iteration, x_new, x_old = args

            # Calculate the update using the backward Euler method
            x_old = x_new
            # x_new = jnp.linalg.solve(jnp.eye(x.shape[0]) - self.dt * self.system(t,x_new), x + self.dt * self.system(t,x_new))
            x_new = jnp.linalg.solve(
                jnp.eye(x.shape[0]) - self.dt * self.system.jacobian(t, x_new) @ (x_new - x),
                x + self.dt * self.system(t, x),
            )
            iteration += 1
            return iteration, x_new, x_old

        def cond(args):
            iteration, x_new, x_old = args
            return (jnp.max(jnp.abs(x_new - x_old)) >= self.tol) & (iteration < self.max_iter)

        # Loop until convergence using lax.while_loop
        iterations, x_new, _ = lax.while_loop(cond, body, (0, x_new, jnp.zeros_like(x_new)))
        return x_new

    def simulate(self, x0: jnp.ndarray, t0: float, tf: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        num_steps = jnp.ceil((tf - t0) / self.dt).astype(int)

        trajectory = jnp.zeros((num_steps + 1, x0.shape[0]))
        trajectory = trajectory.at[0].set(x0)
        times = np.linspace(start=t0, stop=tf, num=num_steps + 1)
        return self._simulate(trajectory, times, num_steps)

    @partial(jit, static_argnums=(0,))
    def _simulate(self, trajectory: jnp.ndarray, times: jnp.ndarray, num_steps: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Create a loop using lax.fori_loop that integrates the system using the backward Euler method and
        # stores the results in the trajectory array
        def body(i, args):
            times, trajectory = args
            t = times[i - 1] + self.dt
            x = trajectory[i - 1]
            trajectory = trajectory.at[i].set(self.step(t, x))
            return times, trajectory

        times, trajectory = lax.fori_loop(1, num_steps + 1, body, (times, trajectory))
        return times, trajectory
