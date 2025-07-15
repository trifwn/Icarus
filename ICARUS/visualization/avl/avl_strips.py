from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from numpy import ndarray
from pandas import DataFrame

from ICARUS.vehicle import Wing
from ICARUS.vehicle import WingSurface
from ICARUS.visualization.utils import validate_surface_input

if TYPE_CHECKING:
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane


def plot_avl_strip_data_3D(
    plane: Airplane,
    state: State,
    case: str,
    surfaces: Sequence[WingSurface | Wing | str] | str | WingSurface | Wing,
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
    from ICARUS.solvers.AVL import get_strip_data

    strip_data = get_strip_data(plane, state, case)
    surface_objects = validate_surface_input(plane, surfaces)

    _surface_names: list[str] = []
    for surface in surface_objects:
        if isinstance(surface, Wing):
            # Get the separate segments of the merged wing
            _surface_names.extend([s.name for s in surface.get_separate_segments()])
            _surface_names.append(surface.name)
        elif isinstance(surface, WingSurface):
            _surface_names.append(surface.name)
        else:
            raise ValueError("surface_name must be a string or a WingSurface object")

    # Get the body data where surface_name is in surface_names
    strip_data = strip_data[strip_data.index.isin(_surface_names)]

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(projection="3d")  # noqa
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
        surf_name = surf_name.strip().replace(" ", "_")
        surf = plane.get_surface(surf_name)

        if surf_name not in strip_data.index:
            continue

        min_j_value: int = int(strip_data[strip_data.index == surf_name]["j"].min())
        max_j_value: int = int(strip_data[strip_data.index == surf_name]["j"].max())

        all_strips = surf.all_strips
        for j, strip in enumerate(all_strips):
            surface_idx = j + min_j_value
            if surface_idx > max_j_value:
                break

            strip_df: DataFrame = strip_data[strip_data["j"] == surface_idx]
            strip_values: list[float] = [
                float(item) for item in strip_df[category].values
            ]
            color: tuple[Any, ...] | ndarray[Any, Any] = cmap(norm(strip_values))
            strip.plot(ax, color=color)

    _ = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.2)
    plt.show()
    return strip_data


def plot_avl_strip_data_2D(
    plane: Airplane,
    state: State,
    case: str,
    surface_name: str | WingSurface,
    category: str = "ai",
) -> DataFrame:
    """
    Function to plot the 2D strips of a given airplane.
    """
    from ICARUS.solvers.AVL import get_strip_data

    if isinstance(surface_name, str):
        surface = plane.get_surface(surface_name)
    elif isinstance(surface_name, WingSurface):
        surface = surface_name
    else:
        raise ValueError("surface_name must be a string or a WingSurface object")

    if isinstance(surface, Wing):
        surface_names = [s.name for s in surface.get_separate_segments()]
        surface_names.append(surface.name)
    else:
        surface_names = [surface.name]
    strip_data = get_strip_data(plane, state, case)

    # Get the Data where strip_data.index in surface_names
    surf_data = strip_data[strip_data.index.isin(surface_names)]

    # Sort the DataFrame by the Yle
    surf_data = surf_data.sort_values("Yle")

    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot()
    ax.set_title(f"{plane.name} {surface_name} {category} Data")
    ax.set_xlabel("Spanwise")
    ax.set_ylabel(category)

    x = surf_data["Yle"]
    y = surf_data[category]
    ax.plot(x, y)
    ax.grid()
    fig.show()
    return strip_data
