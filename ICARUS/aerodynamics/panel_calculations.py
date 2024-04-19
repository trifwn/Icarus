import jax
import jax.numpy as jnp


def induced_vel_calc_vectorized(
    i: int,
    j: int,
    gammas_mat,
    control_points,
    grid,
    solve_fun,
    N: int,
    M: int,
):
    def inner_loop(l):
        Us = jnp.zeros(3)
        Uss = jnp.zeros(3)

        def inner_inner_loop(k):
            U, Ustar = solve_fun(
                control_points[i, j, 0],
                control_points[i, j, 1],
                control_points[i, j, 2],
                l,
                k,
                grid,
                gamma=gammas_mat[l, k],
            )
            return U, Ustar

        Us, Uss = jax.lax.scan(
            lambda acc, k: (acc[0] + inner_inner_loop(k)[0], acc[1] + inner_inner_loop(k)[1]),
            (Us, Uss),
            jnp.arange(M),
        )
        return Us, Uss

    Us, Uss = jax.lax.scan(
        lambda acc, l: (acc[0] + inner_loop(l)[0], acc[1] + inner_loop(l)[1]),
        (jnp.zeros(3), jnp.zeros(3)),
        jnp.arange(N - 1),
    )
    return Us, Uss


def solve_wing_panels(self, Q, solve_fun: Callable[..., tuple[Float[Array, '3'], Float[Array, "3"]]]) -> None:
    self.solve_fun = solve_fun
    RHS_np = self.get_RHS(Q)
    self.RHS_np = RHS_np

    if (self.a_np is not None) and self.wake_geom_type == "TE-Geometrical":
        # print(f"Using previous solution for a and b! Be smart")
        return
    else:
        # print(f"Solving for a and b")
        if self.wake_geom_type == "TE-Geometrical":
            # print("We should be using LU decomposition")
            pass
        a_np, b_np = self.get_LHS(solve_fun)
        self.a_np = a_np
        self.b_np = b_np
