import numpy as np


def ff(num):
    return np.format_float_scientific(num, sign=False, precision=2).zfill(5)


def ff2(num):
    if num >= 0:
        return "{:2.5f}".format(num)
    else:
        return "{:2.4f}".format(num)


def ff3(num):
    if num >= 10:
        return "{:2.5f}".format(num)
    elif num >= 0:
        return "{:2.6f}".format(num)
    else:
        return "{:2.5f}".format(num)


def ff4(num):
    if num >= 0:
        return "{:2.4f}".format(num)
    else:
        return "{:2.3f}".format(num)
