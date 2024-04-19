"""
Module to define the Path class.
TODO: WRITE THE MODULE THIS IS JUST A TEMPLATE
"""

from typing import Any


class Path:
    def __init__(self) -> None:
        """Initialize the Path class."""
        # Flight Path Parameters
        self.flight_path: list[Any] = []
        self.flight_path_time: list[Any] = []
        self.flight_path_altitude: list[Any] = []
        self.flight_path_longitude: list[Any] = []
        self.flight_path_latitude: list[Any] = []
        self.flight_path_speed: list[Any] = []
        self.flight_path_wind: list[Any] = []
        self.flight_path_wind_direction: list[Any] = []

    def set_parameters(
        self,
        positions: list[Any],
        time: list[Any],
        altitudes: list[Any],
        longitudes: list[Any],
        latitudes: list[Any],
        speeds: list[Any],
        winds: list[Any],
        wind_directions: list[Any],
    ) -> None:
        """Set the path parameters."""
        self.flight_path = positions
        self.flight_path_time = time
        self.flight_path_altitude = altitudes
        self.flight_path_longitude = longitudes
        self.flight_path_latitude = latitudes
        self.flight_path_speed = speeds
        self.flight_path_wind = winds
        self.flight_path_wind_direction = wind_directions
