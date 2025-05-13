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

from ICARUS.computation.solvers.AVL.post_process.strips import get_strip_data
from ICARUS.flight_dynamics.state import State
from ICARUS.vehicle.airplane import Airplane
from ICARUS.vehicle.surface import WingSurface


def avl_strips_3d(
    plane: Airplane,
    state: State,
    case: str,
    surface_names: str | list[str] | list[WingSurface],
    category: str = "Wind",
) -> DataFrame:
    """Function to plot the 3D strips of a given airplane.

    Args:
        pln (Airplane): Plane Object
        case (str): Case Name
        NBs (list[int]): List of Surface Body Numbers to plot
        category (str, optional): Category of Data to plot. Defaults to "Wind".

    Returns:
        DataFrame: DataFrame of the strip data

    """
    strip_data = get_strip_data(plane, state, case)

    if isinstance(surface_names, str):
        _surface_names = [surface_names]
    elif isinstance(surface_names, list):
        if all(isinstance(item, WingSurface) for item in surface_names):
            _surface_names = [surf.name for surf in surface_names]

    # Get the body data where surface_name is in surface_names
    strip_data = strip_data[strip_data.index.isin(_surface_names)]

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(projection="3d")  # type: ignore
    ax.set_title(f"{plane.name} {category} Data")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(30, 150)
    ax.axis("scaled")
    ax.set_xlim(-plane.span / 2, plane.span / 2)
    ax.set_ylim(-plane.span / 2, plane.span / 2)
    ax.set_zlim(-plane.span / 2, plane.span / 2)

    max_value: float = strip_data[category].max()
    min_value: float = strip_data[category].min()

    norm = Normalize(vmin=min_value, vmax=max_value)
    cmap: Colormap = cm.get_cmap("viridis", 12)

    for i, surf_name in enumerate(_surface_names):
        surf = plane.get_surface(surf_name)
        print(f"Plotting Body {i + 1}, Name: {surf.name}")

        min_j_value: int = int(strip_data[strip_data.index == surf_name]["j"].min())
        max_j_value: int = int(strip_data[strip_data.index == surf_name]["j"].max())
        symmetric_strips = [strip.return_symmetric() for strip in surf.strips]
        if surf.is_symmetric_y:
            all_strips = [*surf.strips, *symmetric_strips]
        else:
            all_strips = surf.strips
        for j, strip in enumerate(all_strips):
            surface_idx = j + min_j_value
            if surface_idx > max_j_value:
                break

            strip_df: DataFrame = strip_data[strip_data["j"] == surface_idx]
            strip_values: list[float] = [float(item) for item in strip_df[category].values]
            color: tuple[Any, ...] | ndarray[Any, Any] = cmap(norm(strip_values))
            strip.plot(fig, ax, color=color)

    _ = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.2)
    plt.show()
    return strip_data


def avl_strips_2d(
    plane: Airplane,
    state: State,
    case: str,
    surface_name: str | WingSurface,
    category: str = "ai",
) -> DataFrame:
    """
    Function to plot the 2D strips of a given airplane.
    """
    if isinstance(surface_name, str):
        _surface_name = surface_name
    elif isinstance(surface_name, WingSurface):
        _surface_name = surface_name.name

    strip_data = get_strip_data(plane, state, case)
    surf_data = strip_data[strip_data.index == _surface_name]
    # Sort the DataFrame by the Yle
    surf_data = surf_data.sort_values("Yle")

    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot()
    ax.set_title(f"{plane.name} {surface_name} {category} Data")
    ax.set_xlabel("Spanwise")
    # ax.set_ylim(0, 1.1)
    ax.set_ylabel(category)

    x = surf_data["Yle"]
    y = surf_data[category]
    ax.plot(x, y)
    ax.grid()
    fig.show()
    return strip_data
