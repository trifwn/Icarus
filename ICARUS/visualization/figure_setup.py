from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.axes import SubplotBase
from matplotlib.figure import Figure
from numpy import ndarray


def flatten_axes(
    _axs: list[Axes] | ndarray[Any, Any] | Axes | SubplotBase,
) -> list[Axes]:
    if isinstance(_axs, ndarray):
        axs: list[Axes] = _axs.flatten().tolist()
    elif isinstance(_axs, Axes):
        axs = [_axs]
    elif isinstance(_axs, list):
        axs = _axs
    else:
        raise ValueError("Invalid type for axs")
    return axs


def create_subplots(
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool = False,
    sharey: bool = False,
    squeeze: bool = True,
    subplot_kw: dict[str, Any] | None = None,
    gridspec_kw: dict[str, Any] | None = None,
) -> tuple[Figure, list[Axes]]:
    fig = plt.figure()
    _axs = fig.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        subplot_kw=subplot_kw,
        gridspec_kw=gridspec_kw,
    )

    axs = flatten_axes(_axs)
    return fig, axs


def create_single_subplot() -> tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    return fig, ax
