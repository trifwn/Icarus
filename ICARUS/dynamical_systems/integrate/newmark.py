from functools import partial

import jax.numpy as jnp
from jax import jit
from jax import lax

from ..second_order_system import SecondOrderSystem
from .base_integrator import Integrator


class NewmarkIntegrator(Integrator):
    def __init__(
        self,
        dt: float,
        system: SecondOrderSystem,
        gamma: float = 0.5,
        beta: float = 0.25,
    ):
        self.name = "Newmark"
        self.type = "Implicit"
        self.dt = dt
        self.system = system
        self.gamma = gamma
        self.beta = beta

    @partial(jit, static_argnums=(0,))
    def step(
        self,
        u_old: jnp.ndarray,
        v_old: jnp.ndarray,
        a_old: jnp.ndarray,
        t: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        M = self.system.M(t, u_old)
        C = self.system.C(t, u_old)
        K = self.system.f_int(t, u_old)
        f_ext = self.system.f_ext(t, u_old).squeeze()

        # Calculate predictors
        u_new = u_old + self.dt * v_old + (0.5 - self.beta) * self.dt**2 * a_old
        v_new = v_old + (1 - self.gamma) * self.dt * a_old

        # Solve the linear proble Ax = b
        # print the shapes of the matrices
        a_new = jnp.linalg.solve(
            M + self.gamma * self.dt * C + self.beta * self.dt**2 * K,
            f_ext - K @ u_new - C @ v_new,
        )
        # Calculate Correctors
        v_new += self.gamma * self.dt * a_new
        u_new += self.beta * self.dt**2 * a_new
        return u_new, v_new, a_new

    def simulate(
        self,
        x0: jnp.ndarray,
        t0: float,
        tf: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        num_steps = jnp.ceil((tf - t0) / self.dt).astype(int)
        trajectory = jnp.zeros((num_steps + 1, x0.shape[0]))
        trajectory = trajectory.at[0].set(x0)
        times = jnp.linspace(t0, tf, num_steps + 1)  # noqa

        # Split function into two parts
        # return self._simulate(trajectory, times, num_steps)
        # @partial(jit, static_argnums=(0,))
        # def _simulate(self, trajectory: jnp.ndarray, times: jnp.ndarray, num_steps: int) -> tuple[jnp.ndarray, jnp.ndarray]:

        # Simulate the system using the Newmark integrator
        # using lax.fori_loop
        def body(
            i: int,
            args: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            times, trajectory, accelaration = args

            u_old = trajectory[i - 1][: x0.shape[0] // 2].squeeze()
            v_old = trajectory[i - 1][x0.shape[0] // 2 :].squeeze()
            t = times[i - 1]

            u_new, v_new, accelaration = self.step(u_old, v_old, accelaration, t)
            # squeeze the results
            u_new = u_new.squeeze()
            v_new = v_new.squeeze()
            accelaration = accelaration.squeeze()
            x_new = jnp.hstack([u_new, v_new]).squeeze()
            trajectory = trajectory.at[i].set(x_new)

            return times, trajectory, accelaration

        a0 = jnp.zeros_like(x0[: x0.shape[0] // 2]).squeeze()
        times, trajectory, _ = lax.fori_loop(
            1,
            num_steps + 1,
            body,
            (times, trajectory, a0),
        )
        return times, trajectory
