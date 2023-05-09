"""Module for unit conversions and calculations."""


def calc_mach(velocity: float, speed_of_sound: float) -> float:
    return velocity / speed_of_sound


def get_reynolds(velocity: float, char_length: float, viscosity: float) -> float:
    return (velocity * char_length) / viscosity
