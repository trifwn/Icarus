from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import jit
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from .biot_savart import voring

if TYPE_CHECKING:
    from ICARUS.aero import LSPT_Plane


def get_panel_contribution(
    panel: Float[Array, ...],
    control_point: Float[Array, 3],
) -> tuple[Float[Array, 3], Float[Array, 3]]:
    U, Ustar = voring(
        control_point[0],
        control_point[1],
        control_point[2],
        panel,
    )
    return U, Ustar


get_panel_contribution = jit(get_panel_contribution)


def compute_row(
    i: int,
    PANEL_NUM: int,
    panels: Float[Array, ...],
    control_points: Float[Array, ...],
    control_nj: Float[Array, ...],
) -> tuple[Float[Array, ...], Float[Array, ...]]:
    def compute_contributions(
        j: Int[Array, ...],
    ) -> tuple[Float[Array, 3], Float[Array, 3]]:
        # Calculate contributions from all panels
        U, Ustar = get_panel_contribution(
            panels[j],
            control_points[i],
        )
        return jnp.dot(U, control_nj[i]), jnp.dot(Ustar, control_nj[i])

    a_row, b_row = vmap(compute_contributions)(jnp.arange(PANEL_NUM))
    return a_row, b_row


def get_LHS(
    plane: LSPT_Plane,
) -> tuple[Float[Array, ...], Float[Array, ...]]:
    a_rows, b_rows = vmap(
        lambda i: compute_row(
            i,
            plane.PANEL_NUM,
            plane.panels,
            plane.control_points,
            plane.control_nj,
        ),
    )(jnp.arange(plane.PANEL_NUM))
    a_np = jnp.reshape(a_rows, (plane.PANEL_NUM, plane.PANEL_NUM))
    b_np = jnp.reshape(b_rows, (plane.PANEL_NUM, plane.PANEL_NUM))

    # Get the near wake panels row indices
    near_wake_panel_idxs = plane.near_wake_indices
    shedding_panel_idxs = plane.wake_shedding_panel_indices
    a_np = a_np.at[near_wake_panel_idxs, :].set(0)
    a_np = a_np.at[near_wake_panel_idxs, near_wake_panel_idxs].set(1)
    a_np = a_np.at[near_wake_panel_idxs, shedding_panel_idxs].set(-1)

    b_np = b_np.at[near_wake_panel_idxs, :].set(0)
    b_np = b_np.at[near_wake_panel_idxs, near_wake_panel_idxs].set(1)
    b_np = b_np.at[near_wake_panel_idxs, shedding_panel_idxs].set(-1)

    return a_np, b_np


def get_RHS(plane: LSPT_Plane, Q: Array) -> Float[Array, ...]:
    def compute_rhs(i: Int[Array, ...]) -> Float[Array, ...]:
        return -jnp.dot(Q, plane.control_nj[i])

    RHS = vmap(compute_rhs)(jnp.arange(plane.PANEL_NUM))

    # Zero the near wake panels
    near_wake_panel_idxs = plane.near_wake_indices
    RHS = RHS.at[near_wake_panel_idxs].set(0)
    return RHS
