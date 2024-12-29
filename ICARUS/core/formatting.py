import numpy as np


def sps(num: int) -> str:
    """Return a string with the number of spaces specified by num"""
    return " " * num


def tabs(num: int) -> str:
    """Return a string with the number of tabs specified by num"""
    return "\t" * num


def ff(num: float) -> str:
    return str(
        np.format_float_scientific(num, sign=False, precision=3, min_digits=3).zfill(5),
    )


def ff2(num: float) -> str:
    if num >= 0:
        return f"{num:2.5f}"
    return f"{num:2.4f}"


def ff3(num: float) -> str:
    if num >= 10:
        return f"{num:2.5f}"
    if num >= 0:
        return f"{num:2.6f}"
    return f"{num:2.5f}"


def ff4(num: float) -> str:
    if num >= 0:
        return f"{num:1.4e}"
    return f"{num:1.3e}"


def ff5(num: float, n_digits: int = 10) -> str:
    """Given a float, return a string of the number in scientific notation
    that is n_digits characters long in total including the sign and decimal point.
    e.g ff4(0.000000000) ->  "0.0000e+00"
    e.g ff4(0.000000001) ->  "1.0000e-09"
    e.g ff4(1)           ->  "1.0000e+00"
    e.g ff4(10)          ->  "1.0000e+01"
    e.g ff4(100)         ->  "1.0000e+02"
    e.g ff4(538)         ->  "5.3800e+02"
    e.g ff4(-538)        ->  "-5.380e+02"


    Args:
        num (float): Any float
        n_digits (int): Number of characters in the string

    Returns:
        str: The float in scientific notation

    """
    if num == 0 or num > 0:
        return f"{num:1.{n_digits - 7}e}"
    return f"{num:1.{n_digits - 8}e}"
