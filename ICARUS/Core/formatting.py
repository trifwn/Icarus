import numpy as np


def ff(num):
    return np.format_float_scientific(num, sign=False, precision=2).zfill(5)


def ff2(num):
    if num >= 0:
        return f"{num:2.5f}"
    else:
        return f"{num:2.4f}"


def ff3(num):
    if num >= 10:
        return f"{num:2.5f}"
    elif num >= 0:
        return f"{num:2.6f}"
    else:
        return f"{num:2.5f}"


def ff4(num):
    if num >= 0:
        return f"{num:2.4f}"
    else:
        return f"{num:2.3f}"
