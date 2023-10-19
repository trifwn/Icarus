"""Module for unit conversions and calculations."""

si_to_imperial: dict[str, float] = {
    "m" : 3.28084,
    "m2" : 10.7639,
    "kg" : 2.20462,
}

def calc_reynolds(velocity: float, char_length: float, viscosity: float) -> float:
    return (velocity * char_length) / viscosity


def calc_mach(velocity: float, speed_of_sound: float) -> float:
    """Converts speed in m/s to mach number

    Args:
        speed_m_s (float): Speed in m/s

    Returns:
        float: Mach Number of speed
    """

    return velocity / speed_of_sound
