from typing import Any

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from numpy import ndarray
from pandas import DataFrame

from ICARUS.computation.solvers.GenuVP.post_process.strips import get_strip_data
from ICARUS.vehicle.plane import Airplane


def gnvp_strips_3d(
    pln: Airplane,
    case: str,
    NBs: list[int],
    category: str = "Wind",
) -> DataFrame:
    """
    Function to plot the 3D strips of a given airplane.

    Args:
        pln (Airplane): Plane Object
        case (str): Case Name
        NBs (list[int]): List of Surface Body Numbers to plot
        category (str, optional): Category of Data to plot. Defaults to "Wind".

    Returns:
        DataFrame: DataFrame of the strip data
    """
    all_strip_data, body_data = get_strip_data(pln, case, NBs)
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(projection="3d")  # type: ignore
    ax.set_title(f"{pln.name} {category} Data")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(30, 150)
    ax.axis("scaled")
    ax.set_xlim(-pln.span / 2, pln.span / 2)
    ax.set_ylim(-pln.span / 2, pln.span / 2)
    ax.set_zlim(-pln.span / 2, pln.span / 2)
    max_value: float = body_data[category].max()
    min_value: float = body_data[category].min()

    norm = Normalize(vmin=min_value, vmax=max_value)
    cmap: Colormap = cm.get_cmap("viridis", 12)

    for i, wg in enumerate(pln.surfaces):
        if i + 1 not in NBs:
            continue
        for j, surf in enumerate(wg.all_strips):
            strip_df: DataFrame = body_data[(body_data["Body"] == i + 1) & (body_data["Strip"] == j + 1)]

            strip_values: list[float] = [float(item) for item in strip_df[category].values]
            color: tuple[Any, ...] | ndarray[Any, Any] = cmap(norm(strip_values))
            surf.plot(fig, ax, None, color)
    _ = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.2)
    plt.show()
    return all_strip_data


def gnvp_strips_2d(
    pln: Airplane,
    case: str,
    NB: int | list[int],
    category: str = "Wind",
) -> DataFrame | int:
    """
    Plots the 2D strips of a given airplane.

    Args:
        pln (Airplane): Airplane Object
        case (str): Case Name
        NB (int): Surface Body Number
        category (str, optional): Data to visualize. Defaults to "Wind".

    Returns:
        DataFrame | int: Returns the strip data if successful, 0 otherwise
    """
    if isinstance(NB, list):
        if len(NB) >= 1:
            print("Only one body can be selected for 2D plots!")
            print("Selecting First")
            NB = NB[0]

    if type(NB) is not int:
        return 0

    stripDat, data = get_strip_data(pln, case, [NB])
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot()
    ax.set_title(f"{pln.name} {pln.surfaces[NB-1].name} {category} Data")
    ax.set_xlabel("Spanwise")
    # ax.set_ylim(0, 1.1)
    ax.set_ylabel(category)
    x: list[int] = [i for i, data in enumerate(data[category])]
    ax.scatter(x, data[category])

    fig.show()
    plt.show()
    return stripDat
