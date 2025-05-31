from typing import TYPE_CHECKING
import distinctipy
from matplotlib.markers import MarkerStyle

markers_str: list[str | int] = list(MarkerStyle.markers.keys())
markers = [MarkerStyle(marker) for marker in markers_str]

colors_ = distinctipy.get_colors(36)

if TYPE_CHECKING:
    from ICARUS.vehicle import Airplane

def validate_airplane(
    airplanes: list[str | Airplane] | str | Airplane,
) -> tuple[list[str], list[str]]:
    """Validate the airplanes and prefixes."""
    if not isinstance(airplanes, list):
        raise TypeError("airplanes must be a list of Airplane objects or strings")

    if len(airplanes) == 0:
        raise ValueError("No airplanes provided")

    return airplanes

