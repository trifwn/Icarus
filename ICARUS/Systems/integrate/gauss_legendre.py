from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import lax
from jax.debug import print as jprint

from ..base_system import DynamicalSystem
from .base_integrator import Integrator


class GaussLegendreIntegrator(Integrator):
    def __init__(
        self,
        dt: float,
        system: DynamicalSystem,
        tol: float = 1e-6,
        max_iter: int = 1000,
        damping: float = 1,
        n: int = 4,
    ) -> None:
        self.name = "Gauss-Legendre"
        self.type = "Implicit"
        self.dt = dt
        self.system = system
        self.tol = tol
        self.max_iter = max_iter
        # Check damping value
        if damping > 1 or damping <= 0:
            raise ValueError("Damping should be between 0 and 1.")
        self.damping = damping

        # Number of Gauss-Legendre nodes and weights
        self.n = n
        self.nodes, self.weights = np.polynomial.legendre.leggauss(self.n)  # type: ignore

    @partial(jit, static_argnums=(0,))
    def step(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        d = x.shape[0]  # Get the dimension of the state vector

        # Use explicit Euler steps as initial guesses
        k1 = self.system(t, x)

        x1_guess = x + (1 / 2 - jnp.sqrt(3) / 6) * self.dt * k1
        k2 = self.system(t, x1_guess)
        a11 = 1 / 4
        a12 = 1 / 4 - np.sqrt(3) / 6
        a21 = 1 / 4 + np.sqrt(3) / 6
        a22 = 1 / 4

        def error(k1: jnp.ndarray, k2: jnp.ndarray) -> jnp.ndarray:
            """Calculates the error for the current state estimates."""
            predicted_state1 = x + (a11 * self.dt * k1 + a12 * self.dt * k2)
            predicted_state2 = x + (a21 * self.dt * k1 + a22 * self.dt * k2)
            return jnp.concatenate(
                [
                    k1 - self.system(t, predicted_state1),
                    k2 - self.system(t, predicted_state2),
                ],
            )

        def cond_fun(args: tuple[int, jnp.ndarray, jnp.ndarray]) -> bool:
            iteration, k1, k2 = args
            er = error(k1, k2)
            return (jnp.linalg.norm(er) > self.tol) & (iteration < self.max_iter)

        def body_fun(args: tuple[int, jnp.ndarray, jnp.ndarray]) -> tuple[int, jnp.ndarray, jnp.ndarray]:
            iteration, k1, k2 = args
            er = error(k1, k2)
            j1 = self.system.jacobian(t, x + (a11 * self.dt * k1 + a12 * self.dt * k2))
            j2 = self.system.jacobian(t, x + (a21 * self.dt * k1 + a22 * self.dt * k2))

            j = jnp.block(
                [
                    [jnp.eye(d) - self.dt * a11 * j1, -self.dt * a12 * j1],
                    [-self.dt * a21 * j2, jnp.eye(d) - self.dt * a22 * j2],
                ],
            )

            k_next = jnp.concatenate([k1, k2]) - self.damping * jnp.linalg.solve(j, er)
            k1 = k_next[:d]
            k2 = k_next[d:]

            # jprint("Time step {x}, Iteration {i}, Error {e}",x=t, i=iteration, e=jnp.linalg.norm(er))
            return iteration + 1, k1, k2

        iteration, k1, k2 = lax.while_loop(cond_fun, body_fun, (0, k1, k2))
        x_next = x + self.dt / 2 * (k1 + k2)
        return x_next

        # x_guess = [x + self.dt * self.nodes[i] * k1 for i in range(self.n)]
        # ks = [self.system(t + self.dt * self.nodes[i], x_guess[i]) for i in range(self.n)]
        # ks = jnp.array(ks)

        # coefficients = jnp.array([[(self.weights[i] / 2) * (self.dt * (1 + self.nodes[i])) for i in range(self.n)],
        #                           [(self.weights[i] / 2) * (self.dt * (1 - self.nodes[i])) for i in range(self.n)]])

        # def error(ks):
        #     """Calculates the error for the current state estimates."""
        #     predicted_states = [x + sum(coefficients[j][i] * self.dt * ks[i] for i in range(self.n)) for j in range(2)]
        #     return jnp.concatenate([ks[j] - self.system(t, predicted_states[j]) for j in range(2)])

        # def cond_fun(args):
        #     iteration, ks = args
        #     er = error(ks)
        #     return (jnp.linalg.norm(er) > self.tol) & (iteration < self.max_iter)

        # def body_fun(args):
        #     iteration, ks = args
        #     er = error(ks)

        #     # Calculate Jacobian matrix
        #     jacobians = [self.system.jacobian(t, x + jnp.dot(coefficients[j], ks) * self.dt) for j in range(2)]

        #     # Construct the Jacobian matrix
        #     eye_block = jnp.eye(d)
        #     jac_upper = eye_block - jnp.sum(self.dt * coefficients[0][:, None, None] * jacobians[0], axis=0)
        #     jac_lower = -jnp.sum(self.dt * coefficients[1][:, None, None] * jacobians[1], axis=0)

        #     jac = jnp.vstack([jac_upper, jac_lower])

        #     ks_next = ks - self.damping * jnp.linalg.solve(jac, er)

        #     return iteration + 1, ks_next

        # iteration, ks = lax.while_loop(cond_fun, body_fun, (0, ks))

        # # Calculate next state
        # x_next = x + sum(self.dt / 2 * self.weights[i] * (ks[0][i] + ks[1][i]) for i in range(self.n))

        # return x_next

    def simulate(self, x0: jnp.ndarray, t0: float, tf: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        num_steps = jnp.ceil((tf - t0) / self.dt).astype(int)

        trajectory = jnp.zeros((num_steps + 1, x0.shape[0]))
        trajectory = trajectory.at[0].set(x0)
        times = np.linspace(start=t0, stop=tf, num=num_steps + 1)
        return self._simulate(trajectory, times, num_steps)

    @partial(jit, static_argnums=(0,))
    def _simulate(self, trajectory: jnp.ndarray, times: jnp.ndarray, num_steps: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Create a loop using lax.fori_loop that integrates the system using the Gauss-Legendre method and
        # stores the results in the trajectory array
        def body(i, args):
            times, trajectory = args
            t = times[i - 1] + self.dt
            x = trajectory[i - 1]
            trajectory = trajectory.at[i].set(self.step(t, x))
            return times, trajectory

        times, trajectory = lax.fori_loop(1, num_steps + 1, body, (times, trajectory))
        return times, trajectory
