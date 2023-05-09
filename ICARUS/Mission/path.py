class Path:
    def __init__(self) -> None:
        # Flight Path Parameters
        self.flightPath = []
        self.flightPathTime = []
        self.flightPathAltitude = []
        self.flightPathLongitude = []
        self.flightPathLatitude = []
        self.flightPathSpeed = []
        self.flightPathWind = []
        self.flightPathWindDirection = []

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
    ) -> None:
        self.flightPath = positions
        self.flightPathTime = time
        self.flightPathAltitude = altitudes
        self.flightPathLongitude = longitudes
        self.flightPathLatitude = latitudes
        self.flightPathSpeed = speeds
        self.flightPathWind = winds
        self.flightPathWindDirection = windDirections
