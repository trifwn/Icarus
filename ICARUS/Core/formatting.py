import numpy as np


def sps(num: int) -> str:
    """Return a string with the number of spaces specified by num"""
    return " " * num


def tabs(num: int) -> str:
    """Return a string with the number of tabs specified by num"""
    return "\t" * num


def ff(num: float) -> str:
    return str(np.format_float_scientific(num, sign=False, precision=3, min_digits=3).zfill(5))


def ff2(num: float) -> str:
    if num >= 0:
        return f"{num:2.5f}"
    else:
        return f"{num:2.4f}"


def ff3(num: float) -> str:
    if num >= 10:
        return f"{num:2.5f}"
    elif num >= 0:
        return f"{num:2.6f}"
    else:
        return f"{num:2.5f}"


def ff4(num: float) -> str:
    if num >= 0:
        return f"{num:1.4e}"
    else:
        return f"{num:1.3e}"
