from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax import Array

from ICARUS.core.types import FloatArray

if TYPE_CHECKING:
    from ICARUS.aero import LSPT_Plane
    from ICARUS.flight_dynamics import State


def get_potential_loads(
    plane: LSPT_Plane,
    state: State,
    ws: Array,
    gammas: Array,
    verbose: bool = True,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    FloatArray,
    FloatArray,
]:
    dens: float = state.environment.air_density
    umag = state.u_freestream

    L_pan = np.zeros(plane.NM)
    D_pan = np.zeros(plane.NM)
    D_trefftz = 0.0

    for strip in plane.strip_data:
        strip.calc_mean_values()
        g_strips = gammas[strip.panel_idxs]
        w = ws[strip.panel_idxs]

        for j in jnp.arange(0, strip.num_panels - 1):
            if j == 0:
                g = g_strips[j]
            else:
                g = g_strips[j] - g_strips[j - 1]
            L_pan[strip.panel_idxs[j]] = dens * umag * strip.width * g
            D_pan[strip.panel_idxs[j]] = -dens * strip.width * g * w[j]

            if j == strip.num_panels - 1:
                D_trefftz += (
                    -dens / 2 * strip.width * gammas[strip.panel_idxs[j]] * w[j]
                )

    # Calculate the torque. The torque is calculated w.r.t. the CG
    # and is the sum of the torques of each panel times the distance
    # from the CG to the control point of each panel
    M = jnp.array([0, 0, 0], dtype=float)
    for i in jnp.arange(0, plane.NM):
        M += L_pan[i] * jnp.cross(
            plane.control_points[i, :] - plane.CG,
            plane.control_nj[i, :],
        )
        M += D_pan[i] * jnp.cross(
            plane.control_points[i, :] - plane.CG,
            plane.control_nj[i, :],
        )
    Mx, My, Mz = M

    D_pan = D_pan
    L_pan = L_pan
    L: float = float(jnp.sum(L_pan))
    D: float = D_trefftz
    D2: float = float(jnp.sum(D_pan))
    CL: float = 2 * L / (dens * (umag**2) * plane.S)
    CD: float = 2 * D / (dens * (umag**2) * plane.S)
    Cm: float = 2 * My / (dens * (umag**2) * plane.S * plane.MAC)

    if verbose:
        print(f"- Angle {plane.alpha * 180 / jnp.pi}")
        print("\t--Using no penetration condition:")
        print(f"\t\tL:{L}\t|\tD (Trefftz Plane):{D}\tD2:{D2}\t|\tMy:{My}")
        print(f"\t\tCL:{CL}\t|\tCD_ind:{CD}\t|\tCm:{Cm}")

    return L, D, D2, Mx, My, Mz, CL, CD, Cm, L_pan, D_pan


# def get_pseudo_viscous_loads(
#     DB: Database,
#     plane: LSPT_Plane,
#     state: State,
#     verbose: bool = True,
# ):
#     try:
#         (L_2D, D_2D, My_2D, CL_2D, CD_2D, Cm_2D) = integrate_polars_from_reynolds(DB, plane=plane, state=state)
#     except ValueError as e:
#         print("\tCould not interpolate polars! Got error:")
#         print(f"\t{e}")
#         L_2D = 0
#         D_2D = 0
#         My_2D = 0
#         CL_2D = 0
#         CD_2D = 0
#         Cm_2D = 0
#     if verbose:
#         print(f"\t--Using 2D polars:")
#         print(f"\t\tL:{L_2D}\t|\tD:{D_2D}\t|\tMy:{My_2D}")
#         print(f"\t\tCL:{CL_2D}\t|\tCD:{CD_2D}\t|\tCm:{Cm_2D}")
#     return L_2D, D_2D, My_2D, CL_2D, CD_2D, Cm_2D


# def calc_strip_reynolds(
#     plane: LSPT_Plane,
#     state: State,
# ) -> tuple[Float[Array, '...'], Float[Array, '...']]:
#     dens = state.environment.air_density
#     umag = state.u_freestream
#     visc = state.environment.air_dynamic_viscosity

#     for strip in plane.strip_data:
#         pass
#     # Get the effective angle of attack of each strip
#     strips_w_induced = jnp.zeros(plane.num_strips)
#     strips_effective_aoa = jnp.arctan(strips_w_induced / umag) * 180 / jnp.pi + plane.alpha * 180 / jnp.pi

#     # Get the reynolds number of each strip
#     strip_vel = jnp.sqrt(strips_w_induced**2 + umag**2)
#     strip_reynolds = dens * strip_vel * plane.chords / visc

#     # Scan all wing segments and get the orientation of each airfoil
#     # Match that orientation with the each strip and get the effective aoa
#     # That is the angle of attack that the airfoil sees
#     strips_airfoil_effective_aoa = jnp.zeros(plane.num_strips)
#     N: int = 0
#     for wing_seg in plane.surfaces:
#         for j in jnp.arange(0, wing_seg.N - 1):
#             strips_airfoil_effective_aoa[N + j] = strips_effective_aoa[N + j] + wing_seg.orientation[0]
#         N += wing_seg.N - 1

#     return strip_reynolds, strips_airfoil_effective_aoa


# def integrate_polars_from_reynolds(
#     DB: Database,
#     plane: LSPT_Plane,
#     state: State,
#     solver: str = "Xfoil",
# ) -> tuple[Float, Float, Float, Float, Float, Float]:
#     dens: float = state.environment.air_density
#     uinf: float = state.u_freestream

#     strip_CL_2D = jnp.zeros(plane.num_strips)
#     strip_CD_2D = jnp.zeros(plane.num_strips)
#     strip_Cm_2D = jnp.zeros(plane.num_strips)

#     L_2D: float = 0.0
#     D_2D: float = 0.0
#     My_2D: float = 0.0
#     CL_2D: float = 0.0
#     CD_2D: float = 0.0
#     Cm_2D: float = 0.0

#     calc_strip_reynolds(plane, state)

#     N: int = 0
#     L: float = 0
#     D: float = 0
#     My_at_quarter_chord: float = 0
#     for wing_seg in plane.surfaces:
#         airfoil: Airfoil = wing_seg.root_airfoil
#         for j in jnp.arange(0, wing_seg.N - 1):
#             dy: float = float(jnp.mean(plane.grid[N + j + 1, :, 1] - plane.grid[N + j, :, 1]))

#             CL, CD, Cm = DB.foils_db.interpolate_polars(
#                 reynolds=float(strip_reynolds[N + j]),
#                 airfoil_name=airfoil.name,
#                 aoa=float(strip_airfoil_effective_aoa[N + j]),
#                 solver=solver,
#             )
#             strip_CL_2D[N + j] = CL
#             strip_CD_2D[N + j] = CD
#             strip_Cm_2D[N + j] = Cm

#             surface: float = float(plane.chords[N + j]) * dy
#             vel_mag: float = float(jnp.sqrt(plane.w_induced_strips[N + j] ** 2 + uinf**2))
#             dynamic_pressure: float = 0.5 * plane.dens * vel_mag**2

#             # "Integrate" the CL and CD of each strip to get the total L, D and My
#             L += CL * surface * dynamic_pressure
#             D += CD * surface * dynamic_pressure
#             My_at_quarter_chord += Cm * surface * dynamic_pressure * float(plane.chords[N + j])

#         N += wing_seg.N - 1

#     L_2D = L
#     D_2D = D
#     # Calculate Total Moment moving the moment from the quarter chord
#     # to the cg and then add the moment of the lift and drag

#     My_2D = My_at_quarter_chord - D * plane.CG[0] + L * plane.CG[0]

#     CL_2D = 2 * L_2D / (dens * (uinf**2) * plane.S)
#     CD_2D = 2 * D_2D / (dens * (uinf**2) * plane.S)
#     Cm_2D = 2 * My_2D / (dens * (uinf**2) * plane.S * plane.MAC)

#     return L_2D, D_2D, My_2D, CL_2D, CD_2D, Cm_2D
