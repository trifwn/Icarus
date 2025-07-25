from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
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


get_panel_contribution = jax.jit(get_panel_contribution)


def compute_row(
    i: Int[Array, ...],
    panels: Float[Array, ...],
    control_points: Float[Array, ...],
    control_nj: Float[Array, ...],
) -> tuple[Float[Array, ...], Float[Array, ...]]:
    def compute_contributions(
        panel: Float[Array, 3],
    ) -> tuple[Float[Array, 3], Float[Array, 3]]:
        # Calculate contributions from all panels
        U, Ustar = get_panel_contribution(
            panel,
            control_points[i],
        )
        return jnp.dot(U, control_nj[i]), jnp.dot(Ustar, control_nj[i])

    a_row, b_row = vmap(compute_contributions)(panels)
    return a_row, b_row


compute_row = jax.jit(compute_row)


def get_LHS(
    plane: LSPT_Plane,
) -> tuple[Float[Array, ...], Float[Array, ...]]:
    PANEL_NUM = plane.num_panels + plane.num_near_wake_panels

    # Get the Panel data
    contributing_indices = jnp.concatenate(
        [
            plane.panel_indices,
            plane.near_wake_indices,
        ],
        axis=0,
    )

    panels = plane.panels[contributing_indices, :, :]
    panel_cps = plane.panel_cps[contributing_indices, :]
    panel_normals = plane.panel_normals[contributing_indices, :]

    assert PANEL_NUM == panels.shape[0], (
        "Number of panels does not match the number of control points.",
        PANEL_NUM,
        panels.shape[0],
    )

    contribution_fun = partial(
        compute_row,
        panels=panels,
        control_points=panel_cps,
        control_nj=panel_normals,
    )

    a_np, b_np = vmap(contribution_fun)(jnp.arange(PANEL_NUM))

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
    def compute_rhs(panel_normal) -> Float[Array, ...]:
        return -jnp.dot(Q, panel_normal)

    # Get the Panel data
    contributing_indices = jnp.concatenate(
        [
            plane.panel_indices,
            plane.near_wake_indices,
        ],
        axis=0,
    )
    panel_normals = plane.panel_normals[contributing_indices, :]
    RHS = vmap(compute_rhs)(panel_normals)

    # Zero the near wake panels
    near_wake_panel_idxs = plane.near_wake_indices
    RHS = RHS.at[near_wake_panel_idxs].set(0)
    return RHS


def factorize_system(
    plane: LSPT_Plane,
) -> tuple[Array, Array, Array]:
    """
    Factorize the VLM system matrices using LU decomposition.

    Returns:
        Tuple of (A_LU, A_piv, A_star) for efficient solving
    """

    A, A_star = get_LHS(plane)
    # Perform LU decomposition on A
    A_LU, A_piv = jax.scipy.linalg.lu_factor(A)
    return A_LU, A_piv, A_star
