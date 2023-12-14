import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import Collection
from matplotlib.figure import Figure
from numpy import ndarray

from ICARUS.Core.types import ComplexArray


def setup_scatter(
    goal_longitudal: ComplexArray,
    goal_lateral: ComplexArray,
    zero_longitudal: ComplexArray,
    zero_lateral: ComplexArray,
) -> tuple[Figure, ndarray[Any, Any], Collection, Collection]:
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # DUMMY scatters
    zeros = np.zeros(4)

    # Longitudal Mode
    x: list[float] = [ele.real for ele in goal_longitudal]
    y: list[float] = [ele.imag for ele in goal_longitudal]
    axs[0].scatter(x, y, label="Current")
    x = [ele.real for ele in zero_longitudal]
    y = [ele.imag for ele in zero_longitudal]
    axs[0].scatter(x, y, label="Desired")
    collection_long = axs[0].scatter(zeros, zeros, label="Start")
    axs[0].grid()
    axs[0].legend()

    # Lateral Mode
    x = [ele.real for ele in goal_lateral]
    y = [ele.imag for ele in goal_lateral]
    axs[1].scatter(x, y, label="Current")
    x = [ele.real for ele in zero_lateral]
    y = [ele.imag for ele in zero_lateral]
    axs[1].scatter(x, y, label="Desired")
    collection_lat = axs[1].scatter(zeros, zeros, label="Start")
    axs[1].grid()
    axs[1].legend()
    return fig, axs, collection_lat, collection_long


def update_scatter(
    new_eigenvalues_lateral: ComplexArray,
    new_eigenvalues_longitudal: ComplexArray,
    lateral_points: Collection,
    longitudal_points: Collection,
    iteration_num: int,
    fig: Figure,
    axs: ndarray,
) -> None:
    fig.canvas.flush_events()
    title: str = f"Eigenvalues at {iteration_num} of optimization"
    # Make title display the polynomial equation using latex

    fig.suptitle(title, fontsize=16)
    ax_longitudal: Axes = axs[0]
    ax_lateral: Axes = axs[1]

    # scatter Longitudal

    offsets: list[tuple[float, float]] = []
    for item in new_eigenvalues_lateral:
        offsets.append((item.real, item.imag))
    lateral_points.set_offsets(np.array(offsets))

    # scatter Longitudal
    offsets = []
    for item in new_eigenvalues_longitudal:
        offsets.append((item.real, item.imag))
    longitudal_points.set_offsets(np.array(offsets))

    # Relim and Scale
    ax_longitudal.relim()
    ax_longitudal.autoscale()
    ax_lateral.relim()
    ax_lateral.autoscale()

    axs[0].legend(loc="best")
    axs[1].legend()
    fig.canvas.draw()
    time.sleep(0.1)
