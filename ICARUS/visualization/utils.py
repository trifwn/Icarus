from typing import Sequence

import distinctipy
from matplotlib.markers import MarkerStyle

from ICARUS.database import Database
from ICARUS.vehicle import Airplane
from ICARUS.vehicle import Wing
from ICARUS.vehicle import WingSurface

markers_str: list[str | int] = list(MarkerStyle.markers.keys())
markers = [MarkerStyle(marker) for marker in markers_str]

colors_ = distinctipy.get_colors(36)


def get_distinct_markers(n: int) -> list[MarkerStyle]:
    """Get a list of distinct markers."""
    if n > len(markers):
        raise ValueError(f"Requested {n} markers, but only {len(markers)} are available.")
    return markers[:n]


def get_distinct_colors(n: int) -> list[tuple[float, float, float]]:
    """Get a list of distinct colors."""
    colors = distinctipy.get_colors(n)
    return colors


def validate_airplane_input(
    airplanes: list[str | Airplane] | str | Airplane,
) -> list[Airplane]:
    if isinstance(airplanes, str) or isinstance(airplanes, Airplane):
        _airplanes = [airplanes]
    elif isinstance(airplanes, list):
        _airplanes = airplanes

    if not isinstance(_airplanes, list):
        raise TypeError(
            f"Invalid type for airplanes: {type(airplanes)}. Expected str, Airplane, or list of str/Airplane.",
        )

    airplanes_objs: list[Airplane] = validate_airplanes(_airplanes)
    return airplanes_objs


def validate_airplanes(
    airplanes: list[str | Airplane],
    DB: Database | None = None,
) -> list[Airplane]:
    """Validate the airplanes and prefixes."""

    airplane_objects = []
    for airplane in airplanes:
        airplane_objects.append(validate_airplane(airplane, DB))
    return airplane_objects


def validate_airplane(
    airplane: str | Airplane,
    DB: Database | None = None,
) -> Airplane:
    """Validate a single airplane."""

    if isinstance(airplane, str):
        if DB is None:
            DB = Database.get_instance()
        return DB.get_vehicle(airplane)
    elif isinstance(airplane, Airplane):
        return airplane
    else:
        raise TypeError(f"Invalid type for airplane: {type(airplane)}. Expected str or Airplane.")


def validate_surface_input(
    plane,
    surfaces: Sequence[WingSurface | Wing | str] | str | WingSurface | Wing,
) -> list[WingSurface | Wing]:
    if isinstance(surfaces, str):
        surface_objects: list[WingSurface | Wing] = [plane.get_surface(surfaces)]
    elif isinstance(surfaces, WingSurface):
        surface_objects = [surfaces]
    elif isinstance(surfaces, Wing):
        surface_objects = [surfaces]
    elif isinstance(surfaces, list):
        surface_objects: list[WingSurface | Wing] = []
        for item in surfaces:
            if isinstance(item, str):
                surface_objects.append(plane.get_surface(item))
            elif isinstance(item, WingSurface) or isinstance(item, Wing):
                surface_objects.append(item)
            else:
                raise ValueError("surfaces must be a string, WingSurface, or Wing object")
    return surface_objects
