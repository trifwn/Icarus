import numpy as np


def trimState(plane):
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
    print(plane.polars3D.keys())
    trimLoc1 = np.argmin(np.abs(plane.polars3D["Cm"]))
    # Find the polar that is closest to the trim but positive
    trimLoc2 = trimLoc1
    if plane.polars3D["Cm"][trimLoc1] < 0:
        while plane.polars3D["Cm"][trimLoc2] < 0:
            trimLoc2 -= 1
    else:
        while plane.polars3D["Cm"][trimLoc2] > 0:
            trimLoc2 += 1

    # from trimLoc1 and trimLoc2, interpolate the angle where Cm = 0
    dCm = plane.polars3D["Cm"][trimLoc2] - plane.polars3D["Cm"][trimLoc1]
    dAoA = plane.polars3D["AoA"][trimLoc2] - plane.polars3D["AoA"][trimLoc1]

    AoA_trim = plane.polars3D["AoA"][trimLoc1] - \
        plane.polars3D["Cm"][trimLoc1] * dAoA / dCm

    Cm_trim = plane.polars3D["Cm"][trimLoc1] + \
        (plane.polars3D["Cm"][trimLoc2] - plane.polars3D["Cm"][trimLoc1]) * \
        (AoA_trim - plane.polars3D["AoA"][trimLoc1]) / \
        (plane.polars3D["AoA"][trimLoc2] - plane.polars3D["AoA"][trimLoc1])

    CL_trim = plane.polars3D["CL"][trimLoc1] + \
        (plane.polars3D["CL"][trimLoc2] - plane.polars3D["CL"][trimLoc1]) * \
        (AoA_trim - plane.polars3D["AoA"][trimLoc1]) / \
        (plane.polars3D["AoA"][trimLoc2] - plane.polars3D["AoA"][trimLoc1])

    # Print How accurate is the trim
    print(
        f"Cm is {plane.polars3D['Cm'][trimLoc1]} instead of 0 at AoA = {plane.polars3D['AoA'][trimLoc1]}")
    print(
        f"Interpolated values are: AoA = {AoA_trim} , Cm = {Cm_trim}, Cl = {CL_trim}")

    # Find the trim velocity
    S = plane.pln.S
    dens = plane.pln.dens
    W = plane.pln.M * 9.81
    U_cruise = np.sqrt(W / (0.5 * dens * CL_trim * S))
    print(f"Trim velocity is {U_cruise} m/s")
    trim = {
        "U": U_cruise,
        "AoA": AoA_trim,
    }
    return trim
