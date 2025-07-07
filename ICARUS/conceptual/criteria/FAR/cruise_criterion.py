import numpy as np

from ICARUS.core.types import FloatArray


# FAR 5 -> 25.115
def far_5_cruise_speed(
    wing_loading: FloatArray,
    altitude: float,
    MACH: float,
    cd_0: float,
    AR: float,
    cl: float,
    e: float,
) -> tuple[FloatArray, FloatArray]:
    """Returns the thrust loading for a given wing loading, altitude, MACH, cd_0, AR, cl and e

    Args:
        wing_loading (float): Wing loading
        altitude (float): Altitude
        MACH (float): Mach number
        cd_0 (float): Drag coefficient at zero lift
        AR (float): Aspect ratio
        cl (float): Cl at cruise
        e (float): Oswald efficiency factor with compressibility correction

    Returns:
        _type_: Thrust loading

    """
    q_over_mach_sq = (
        1478.36
        - 0.0523741 * altitude
        + 6.90146 * (10**-7) * altitude**2
        - 3.32065 * (10**-12) * altitude**3
    )
    q: float = q_over_mach_sq * MACH**2

    thrust_loading = q * (cd_0 + 0.005) / wing_loading + wing_loading / (
        np.pi * AR * e * q
    )
    return (wing_loading, thrust_loading)
