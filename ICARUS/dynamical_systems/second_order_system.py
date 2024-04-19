from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import jit
from numpy import ndarray

from ICARUS.core.types import FloatArray

from .base_system import DynamicalSystem
from .first_order_system import NonLinearSystem


class SecondOrderSystem(DynamicalSystem):
    """
    This class is for second order systems of the form:
    M(u,t) u'' + C(u,t) u' + f_int(u,t) u = f_ext(u,t)
    """

    def __init__(
        self,
        M: Callable[[float, jnp.ndarray], jnp.ndarray] | jnp.ndarray | FloatArray,
        K: Callable[[float, jnp.ndarray], jnp.ndarray] | jnp.ndarray | FloatArray,
        C: Callable[[float, jnp.ndarray], jnp.ndarray] | jnp.ndarray | FloatArray,
        f_ext: Callable[[float, jnp.ndarray], jnp.ndarray] | jnp.ndarray | FloatArray = lambda t, u: jnp.zeros_like(
            u,
        ),
    ) -> None:
        if isinstance(M, jnp.ndarray) or isinstance(M, ndarray):
            if M.shape == () or M.shape == (1,):
                M = M.reshape(-1, 1)
            self.M = jit(lambda t, x: M)
        else:
            self.M = jit(M)

        if isinstance(C, jnp.ndarray) or isinstance(C, ndarray):
            if C.shape == () or C.shape == (1,):
                C = C.reshape(-1, 1)
            self.C = jit(lambda t, x: C)
        else:
            self.C = jit(C)

        if isinstance(K, jnp.ndarray) or isinstance(K, ndarray):
            if K.shape == () or K.shape == (1,):
                K = K.reshape(-1, 1)
            self.f_int = jit(lambda t, x: K)
        else:
            self.f_int = jit(K)

        if isinstance(f_ext, ndarray):
            f_ext = jnp.array(f_ext)

        if isinstance(f_ext, jnp.ndarray):
            if f_ext.shape == () or f_ext.shape == (1,):
                f_ext = f_ext.reshape(-1, 1)
            f_ext_fun: Callable[[float, jnp.ndarray], jnp.ndarray] = lambda t, x: f_ext
        else:
            f_ext_fun = f_ext

        # Wrap F_ext in order to make sure it returns a column vector of the same size as u
        def f_ext_wrapped(t: float, x: jnp.ndarray) -> jnp.ndarray:
            return f_ext_fun(t, x).reshape(-1, 1)

        # self.f_ext = jit(f_ext_wrapped)
        self.f_ext = f_ext_wrapped

    @partial(jit, static_argnums=(0,))
    def jacobian(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jax.jacobian(self.__call__, argnums=1)(t, x)

    def convert_to_first_order(self) -> NonLinearSystem:
        return NonLinearSystem(f=self.__call__)

    def eigenvalues(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        A, _ = self.linearize(t, x)
        return jnp.linalg.eigvals(A)

    @partial(jit, static_argnums=(0,))
    def __call__(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        # Convert system to a first order system of ODEs by defining the state as [u, v]
        # Then the output is given by:
        # x' = [

        #           v,
        #           M^-1 @ (f_ext - f_int @ u - C @ v)
        #      ]
        # where u is the displacement and v is the velocity of the system
        # and M is the mass matrix, C is the damping matrix, f_int is the internal force and f_ext is the external force
        # The mass matrix M, damping matrix C, internal force f_int and external force f_ext are all functions of time and the state x
        # The state x is a column vector of size 2n where n is the size of the state
        # The first n elements of x represent the displacement u and the second n elements represent the velocity v

        n = x.shape[0] // 2  # Half the size for each part of the state
        u = x[:n].reshape(-1, 1).squeeze()  # First half represents u (displacement), reshape to column vector
        v = x[n:].reshape(-1, 1).squeeze()  # Second half represents v (velocity), reshape to column vector

        M = self.M(t, x)
        C = self.C(t, x)
        f_int = self.f_int(t, x)
        f_ext = self.f_ext(t, x).squeeze()

        # If the mass matrix is a scalar, then we can't use the inverse
        if isinstance(M, jnp.ndarray):
            if M.shape == () or M.shape == (1,) or M.shape == (1, 1):
                a = (f_ext - f_int * u - C * v) / M  # Solve for the acceleration
                a = a.reshape(-1, 1).squeeze()
                return jnp.concatenate([v, a], axis=0).squeeze()
            else:
                a = jnp.linalg.solve(M, f_ext - f_int @ u - C @ v).squeeze()  # Solve for the acceleration
        else:
            raise ValueError("Mass matrix M must be a square matrix")
        return jnp.concatenate([v, a], axis=0).squeeze()

    @partial(jit, static_argnums=(0,))
    def linearize(self, t: float, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        A = self.jacobian(t, x)
        B = self(t, x) - A.squeeze() @ x
        return A, B
