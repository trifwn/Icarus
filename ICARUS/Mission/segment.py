from typing import Any

from .path import Path


class MissionSegment:
    def __init__(self, name: str, type: str, path: Path | None = None) -> None:
        # Mission Segment Definition
        self.name: str = name
        self.type: str = type

        # Mission Start Position Parameters
        self.startingTime: float = 0
        self.startingLatitude: float = 0
        self.startingLongitude: float = 0
        self.startingAltitude: float = 0
        self.startingSpeed: float = 0
        self.startingWind: float = 0
        self.startingWindDirection: float = 0

        # Mission End Position Parameters
        self.endingTime: float = 0
        self.endingLatitude: float = 0
        self.endingLongitude: float = 0
        self.endingAltitude: float = 0
        self.endingSpeed: float = 0
        self.endingWind: float = 0
        self.endingWindDirection: float = 0

        if isinstance(path, Path):
            self.path: Path = path
        else:
            self.path = Path()

    def set_starting_pos(
        self,
        time: float,
        latitude: float,
        longitude: float,
        altitude: float,
        speed: float,
        wind_speed: float,
        windDirection: float,
    ) -> None:
        self.startingTime = time
        self.startingLatitude = latitude
        self.startingLongitude = longitude
        self.startingAltitude = altitude
        self.startingSpeed = speed
        self.startingWind = wind_speed
        self.startingWindDirection = windDirection

    def set_ending_pos(
        self,
        time: float,
        latitude: float,
        longitude: float,
        altitude: float,
        speed: float,
        wind_speed: float,
        windDirection: float,
    ) -> None:
        self.endingTime = time
        self.endingLatitude = latitude
        self.endingLongitude = longitude
        self.endingAltitude = altitude
        self.endingSpeed = speed
        self.endingWind = wind_speed
        self.endingWindDirection = windDirection

    def set_path(self, path: Path) -> None:
        self.path = path

    def set_path_parameters(
        self,
        positions: list[Any],
        time: list[Any],
        altitudes: list[Any],
        longitudes: list[Any],
        latitudes: list[Any],
        speeds: list[Any],
        winds: list[Any],
        windDirections: list[Any],
    ) -> None:
        self.path.set_parameters(
            positions,
            time,
            altitudes,
            longitudes,
            latitudes,
            speeds,
            winds,
            windDirections,
        )
