import numpy as np


def calc_mach(v, c):
    return v / c


def Re(v, c, n):
    return (v * c) / n
