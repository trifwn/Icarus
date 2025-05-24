from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pandas as pd
from pandas import DataFrame

from ICARUS.core.types import FloatArray
from ICARUS.flight_dynamics.state import State

if TYPE_CHECKING:
    from ICARUS.flight_dynamics import State
    from ICARUS.aero.plane_lspt import LSPT_Plane

from ..post_process import get_potential_loads
from . import get_LHS
from . import get_RHS


def aseq(
    plane: LSPT_Plane,
    angles: FloatArray | list[float],
    state: State,
) -> DataFrame:
    # plane.factorize_system()
    A_LU, A_piv, A_star = factorize_system(plane)
    umag = state.u_freestream
    # dens = state.environment.air_density

    Ls = jnp.zeros(len(angles))
    Ds = jnp.zeros(len(angles))
    Mys = jnp.zeros(len(angles))
    CLs = jnp.zeros(len(angles))
    CDs = jnp.zeros(len(angles))
    Cms = jnp.zeros(len(angles))
    Ls_2D = jnp.zeros(len(angles))
    Ds_2D = jnp.zeros(len(angles))
    Mys_2D = jnp.zeros(len(angles))
    # CLs_2D = jnp.zeros(len(angles))
    # CDs_2D = jnp.zeros(len(angles))
    # Cms_2D = jnp.zeros(len(angles))

    for i, aoa in enumerate(angles):
        plane.alpha = aoa * jnp.pi / 180
        plane.beta = 0

        Uinf = umag * jnp.cos(plane.alpha) * jnp.cos(plane.beta)
        Vinf = umag * jnp.cos(plane.alpha) * jnp.sin(plane.beta)
        Winf = umag * jnp.sin(plane.alpha) * jnp.cos(plane.beta)

        Q = jnp.array((Uinf, Vinf, Winf))
        RHS = get_RHS(plane, Q)

        gammas = jax.scipy.linalg.lu_solve((A_LU, A_piv), RHS)
        plane.gammas = gammas
        w = jnp.matmul(plane.A_star, gammas)

        # strips_w_induced = jnp.zeros(len(self.strips))
        # strips_gammas = jnp.zeros(len(self.strips))
        for strip in plane.strip_data:
            strip_idxs = strip.panel_idxs
            strip.gammas = gammas[strip_idxs]
            strip.w_induced = w[strip_idxs]
            strip.calc_mean_values()

        (L, D, D2, Mx, My, Mz, CL, CD, Cm, L_pan, D_pan) = get_potential_loads(
            plane=plane,
            state=state,
            ws=w,
            gammas=gammas,
        )
        # Store the results
        plane.L_pan = L_pan
        plane.D_pan = D_pan

        # No pen
        Ls = Ls.at[i].set(L)
        Ds = Ds.at[i].set(D)
        Mys = Mys.at[i].set(My)
        CLs = CLs.at[i].set(CL)
        CDs = CDs.at[i].set(CD)
        Cms = Cms.at[i].set(Cm)

        if True:
            Ls = Ls.at[i].set(2 * Ls[i])
            Ds = Ds.at[i].set(2 * Ds[i])
            Mys = Mys.at[i].set(2 * Mys[i])
            CLs = CLs.at[i].set(2 * CLs[i])
            CDs = CDs.at[i].set(2 * CDs[i])
            Cms = Cms.at[i].set(2 * Cms[i])

        # 2D polars
        # L_2D, D_2D, My_2D, CL_2D, CD_2D, Cm_2D =
        # Ls_2D[i] = L_2D
        # Ds_2D[i] = D_2D
        # Mys_2D[i] = My_2D
        # CLs_2D[i] = CL_2D
        # CDs_2D[i] = CD_2D
        # Cms_2D[i] = Cm_2D

    df = pd.DataFrame(
        {
            "AoA": angles,
            "LSPT Potential Fz": Ls,
            "LSPT Potential Fx": Ds,
            "LSPT Potential My": Mys,
            "LSPT 2D Fz": Ls_2D,
            "LSPT 2D Fx": Ds_2D,
            "LSPT 2D My": Mys_2D,
        },
    )
    return df


def factorize_system(
    plane: LSPT_Plane,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    A, A_star = get_LHS(plane)
    # Perform LU decomposition on A and A_star
    A_LU, A_piv = jax.scipy.linalg.lu_factor(A)
    # A_star_LU, A_star_piv = jax.scipy.linalg.lu_factor(A_star)
    # self.A_LU = A_LU
    # self.A_piv = A_piv
    # self.A_star = A_star
    return A_LU, A_piv, A_star
