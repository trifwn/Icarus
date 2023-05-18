from .path import Path


class MissionSegment:
    def __init__(self, name: str, type, path=None):

        # Mission Segment Definition
        self.name = name
        self.type = type

        # Mission Start Position Parameters
        self.startingTime = 0
        self.startingLatitude = 0
        self.startingLongitude = 0
        self.startingAltitude = 0
        self.startingSpeed = 0
        self.startingWind = 0
        self.startingWindDirection = 0

        # Mission End Position Parameters
        self.endingTime = 0
        self.endingLatitude = 0
        self.endingLongitude = 0
        self.endingAltitude = 0
        self.endingSpeed = 0
        self.endingWind = 0
        self.endingWindDirection = 0
        self.endingWindDirection = 0

        if path is None:
            self.path = Path()
        else:
            self.path = path

    def setStartingPosition(
        self,
        time,
        latitude,
        longitude,
        altitude,
        speed,
        wind,
        windDirection,
    ):
        self.startingTime = time
        self.startingLatitude = latitude
        self.startingLongitude = longitude
        self.startingAltitude = altitude
        self.startingSpeed = speed
        self.startingWind = wind
        self.startingWindDirection = windDirection

    def setEndingPosition(
        self,
        time,
        latitude,
        longitude,
        altitude,
        speed,
        wind,
        windDirection,
    ):
        self.endingTime = time
        self.endingLatitude = latitude
        self.endingLongitude = longitude
        self.endingAltitude = altitude
        self.endingSpeed = speed
        self.endingWind = wind
        self.endingWindDirection = windDirection

    def setPath(self, path):
        self.path = path

    def setPathParameters(
        self,
        positions,
        time,
        altitudes,
        longitudes,
        latitudes,
        speeds,
        winds,
        windDirections,
    ):
        self.path.setPathParameters(
            positions,
            time,
            altitudes,
            longitudes,
            latitudes,
            speeds,
            winds,
            windDirections,
        )
