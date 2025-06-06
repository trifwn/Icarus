"""Trim module"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ICARUS.flight_dynamics import State


class TrimNotPossible(Exception):
    """Raise when Trim is not Possible due to Negative CL At trim angle"""


class TrimOutsidePolars(Exception):
    """Raise when Trim can't be computed due to Cm not crossing zero at the imported polars"""


def trim_state(state: State, verbose: bool = True) -> dict[str, float]:
    """This function returns the trim conditions of the airplane
    It is assumed that the airplane is trimmed at a constant altitude
    The trim conditions are:
    - Velocity
    - Angle of attack
    - Angle of sideslip         ! NOT IMPLEMENTED YET
    - Elevator deflection       ! NOT IMPLEMENTED YET
    - Aileron deflection        ! NOT IMPLEMENTED YET
    - Rudder deflection         ! NOT IMPLEMENTED YET
    - Throttle setting          ! NOT IMPLEMENTED YET
    - Engine torque             ! NOT IMPLEMENTED YET
    - Engine power              ! NOT IMPLEMENTED YET
    - Engine thrust             ! NOT IMPLEMENTED YET
    - Engine fuel flow          ! NOT IMPLEMENTED YET
    - Engine fuel consumption   ! NOT IMPLEMENTED YET
    - Engine fuel remaining     ! NOT IMPLEMENTED YET
    """
    # Find the index of the closest positive value to zero
    Cm = state.polar["Cm"]
    try:
        trim_loc1: int = int((Cm[Cm >= 0] - 0).idxmin())
        # Find the index of the closest negative value to zero
        trim_loc2: int = int((-Cm[Cm < 0] - 0).idxmin())
    except ValueError as e:
        logging.debug(
            "Trim not possible due to Cm not crossing zero at the imported polars",
        )
        logging.debug(f"Gave error: {e}")
        raise TrimOutsidePolars

    # from trimLoc1 and trimLoc2, interpolate the angle where Cm = 0
    d_aoa = state.polar["AoA"][trim_loc2] - state.polar["AoA"][trim_loc1]
    d_cm = state.polar["Cm"][trim_loc2] - state.polar["Cm"][trim_loc1]
    d_cl = state.polar["CL"][trim_loc2] - state.polar["CL"][trim_loc1]
    d_cd = state.polar["CD"][trim_loc2] - state.polar["CD"][trim_loc1]

    if trim_loc1 < trim_loc2:
        if trim_loc1 != 0:
            trim_loc3 = trim_loc1 - 1
            d2_cd = state.polar["CD"][trim_loc3] - 2 * state.polar["CD"][trim_loc1] + state.polar["CD"][trim_loc2]
        else:
            d2_cd = 0
    elif trim_loc1 != len(state.polar["CD"]) - 1:
        trim_loc3 = trim_loc1 + 1
        d2_cd = state.polar["CD"][trim_loc3] - 2 * state.polar["CD"][trim_loc1] + state.polar["CD"][trim_loc2]
    else:
        d2_cd = 0

    aoa_trim = state.polar["AoA"][trim_loc1] - state.polar["Cm"][trim_loc1] * d_aoa / d_cm

    cm_trim = state.polar["Cm"][trim_loc1] + (aoa_trim - state.polar["AoA"][trim_loc1]) * d_cm / d_aoa
    cl_trim = state.polar["CL"][trim_loc1] + (aoa_trim - state.polar["AoA"][trim_loc1]) * d_cl / d_aoa
    cd_trim = (
        state.polar["CD"][trim_loc1]
        + (aoa_trim - state.polar["AoA"][trim_loc1]) * d_cd / d_aoa
        + (aoa_trim - state.polar["AoA"][trim_loc1]) ** 2 * d2_cd / (d_aoa**2)
    )

    if cl_trim <= 0:
        raise TrimNotPossible
    # Find the trim velocity
    S: float = state.S
    dens: float = state.environment.air_density
    W: float = state.mass * 9.81
    U_CRUISE: float = np.sqrt(W / (0.5 * dens * cl_trim * S))
    CL_OVER_CD = cl_trim / cd_trim
    CM0: float = float(state.polar[state.polar["AoA"] == 0.0]["Cm"].to_list()[0])

    # Print How accurate is the trim
    if verbose:
        print(
            f"Cm is {state.polar['Cm'][trim_loc1]} instead of 0 at AoA = {state.polar['AoA'][trim_loc1]}",
        )
        print(
            f"Interpolated values are: AoA = {aoa_trim} , Cm = {cm_trim}, Cl = {cl_trim}",
        )
        print(f"Trim velocity is {U_CRUISE} m/s")

    # Calculate the static margin
    aoas = state.polar["AoA"].to_numpy(dtype=float)
    cls = state.polar["CL"].to_numpy(dtype=float)
    cms = state.polar["Cm"].to_numpy(dtype=float)
    mac = state.mean_aerodynamic_chord
    x_cg: float = state.CG[0]

    def movement(new_cg: float) -> float:
        new_CM = cms + (new_cg - x_cg) * (cls) / mac

        # New trim location
        cm_sorted_idx = np.argsort(new_CM)
        aoa_trim = np.interp(0, new_CM[cm_sorted_idx], aoas[cm_sorted_idx])

        # Get the slope of the curve near the trim location
        slopes = np.gradient(new_CM, aoas)
        slope = float(np.interp(aoa_trim, aoas, slopes))
        return slope

    # Find where the slope is zero
    from scipy.optimize import fsolve

    x_neutral_point = fsolve(movement, float(x_cg), xtol=1e-5)[0]
    static_margin = (x_neutral_point - x_cg) / mac

    trim: dict[str, float] = {
        "U": U_CRUISE,
        "AoA": aoa_trim,
        "CL": cl_trim,
        "CD": cd_trim,
        "CL/CD": CL_OVER_CD,
        "Cm0": CM0,
        "Static Margin": static_margin,
        "Neutral Point": x_neutral_point,
    }
    return trim

    # _trim = Trim(
    #     velocity=U_CRUISE,
    #     angle_of_attack=aoa_trim,
    #     cl=cl_trim,
    #     cd=cd_trim,
    #     cl_over_cd=CL_OVER_CD,
    #     cm0=CM0,
    #     static_margin=static_margin,
    #     neutral_point=x_neutral_point,
    # )


class Trim:
    """Trim class"""

    def __init__(
        self,
        velocity: float,
        angle_of_attack: float,
        cl: float,
        cd: float,
        cl_over_cd: float,
        cm0: float,
        static_margin: float,
        neutral_point: float,
    ) -> None:
        self.velocity = velocity
        self.angle_of_attack = angle_of_attack
        self.cl = cl
        self.cd = cd
        self.cl_over_cd = cl_over_cd
        self.cm0 = cm0
        self.static_margin = static_margin
        self.neutral_point = neutral_point
