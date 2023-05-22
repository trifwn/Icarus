"""
Trim module
"""
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ICARUS.Flight_Dynamics.state import State


def trim_state(state: "State") -> dict[str, float]:
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

    # Index of interest in the Polar Dataframe
    print(state.polars.keys())
    trim_loc1 = np.argmin(np.abs(state.polars["Cm"]))
    # Find the polar that is closest to the trim but positive
    trim_loc2 = trim_loc1
    if state.polars["Cm"][trim_loc1] < 0:
        while state.polars["Cm"][trim_loc2] < 0:
            trim_loc2 -= 1
    else:
        while state.polars["Cm"][trim_loc2] > 0:
            trim_loc2 += 1

    # from trimLoc1 and trimLoc2, interpolate the angle where Cm = 0
    d_cm = state.polars["Cm"][trim_loc2] - state.polars["Cm"][trim_loc1]
    d_aoa = state.polars["AoA"][trim_loc2] - state.polars["AoA"][trim_loc1]

    aoa_trim = (
        state.polars["AoA"][trim_loc1] - state.polars["Cm"][trim_loc1] * d_aoa / d_cm
    )

    cm_trim = state.polars["Cm"][trim_loc1] + (
        state.polars["Cm"][trim_loc2] - state.polars["Cm"][trim_loc1]
    ) * (aoa_trim - state.polars["AoA"][trim_loc1]) / (
        state.polars["AoA"][trim_loc2] - state.polars["AoA"][trim_loc1]
    )

    cl_trim = state.polars["CL"][trim_loc1] + (
        state.polars["CL"][trim_loc2] - state.polars["CL"][trim_loc1]
    ) * (aoa_trim - state.polars["AoA"][trim_loc1]) / (
        state.polars["AoA"][trim_loc2] - state.polars["AoA"][trim_loc1]
    )

    # Print How accurate is the trim
    print(
        f"Cm is {state.polars['Cm'][trim_loc1]} instead of 0 at AoA = {state.polars['AoA'][trim_loc1]}",
    )
    print(f"Interpolated values are: AoA = {aoa_trim} , Cm = {cm_trim}, Cl = {cl_trim}")

    # Find the trim velocity
    S: float = state.S
    dens: float = state.env.air_density
    W: float = state.mass * 9.81
    U_CRUISE: float = np.sqrt(W / (0.5 * dens * cl_trim * S))
    print(f"Trim velocity is {U_CRUISE} m/s")
    trim: dict[str, float] = {
        "U": U_CRUISE,
        "AoA": aoa_trim,
    }
    return trim
