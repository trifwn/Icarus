import numpy as np


class Environment:
    def __init__(self, name: str) -> None:
        """Definition of the Environment object

        # !TODO: IMPLEMENT https://pypi.org/project/ambiance/

        Args:
            name (str): Name of the environment

        """
        self.name: str = name
        self.GRAVITY: float = 9.81

        # Atmospheric properties (default values == sea level == reference conditions)
        self.altitude: float = 0
        self.temperature: float = 15 + 273.15
        self.pressure: float = 101325

        # Reference properties
        self.reference_altitude: float = 0
        self.reference_temperature: float = 15 + 273.15
        self.reference_pressure: float = 101325

        # Thermodynamic properties
        self.air_molar_mass: float = 0.0289644
        self.air_specific_heat_constant_pressure: float = 1005.7
        self.UNIVERSAL_GAS_CONSTANT: float = 8.31447
        self.SPECIFIC_GAS_CONSTANT: float = self.UNIVERSAL_GAS_CONSTANT / self.air_molar_mass
        self.gamma: float = 1.4
        self.air_specific_heat_constant_volume: float = 718.7

        # Flow properties
        self.air_speed_of_sound: float = self.get_speed_of_sound(
            self.gamma,
            self.SPECIFIC_GAS_CONSTANT,
            self.temperature,
        )

        # Thermal properties
        self.air_thermal_conductivity: float = 0.0257
        self.air_thermal_diffusivity: float = 1.25e-5
        self.air_thermal_expansion_coefficient: float = 0.000012
        self.air_prandtl_number: float = 0.71

    def _set_altitude(self, h: float, units: str = "SI") -> None:
        """Set the altitude of the environment

        Args:
            h (float): Altitude in meters or feet
            units (str, optional): Defaults to 'SI'.

        """
        if units == "Imperial":
            h *= 0.3048
        self.altitude = h

    def _set_temperature_from_altitude(self, h: float) -> None:
        """Reference from https://www.omnicalculator.com/physics/altitude-temperature

        Args:
            h (float): Altitude in meters

        """
        if h < 11000:
            self.temperature = self.reference_temperature + 15.04 - 0.00649 * h
        elif h < 15000:
            self.temperature = self.reference_temperature - 56.46
        elif h < 51000:
            self.temperature = self.reference_temperature - 56.46 + 0.0015 * (h - 15000)
        elif h < 86000:
            self.temperature = self.reference_temperature - 2 + 0.0025444 * (h - 51000)
        else:
            self.temperature = self.reference_temperature - 87.05

    def _set_pressure_from_altitude_and_temperature(self, h: float, T: float) -> None:
        """Reference from https://www.omnicalculator.com/physics/air-pressure-at-altitude

        Args:
            h (float): Altitude in meters
            T (float): Temperature in Kelvin

        """
        self.temperature = T
        self._set_altitude(h)

        self.pressure = self.reference_pressure * np.exp(
            -self.GRAVITY * self.air_molar_mass * (h - self.reference_altitude) / (self.UNIVERSAL_GAS_CONSTANT * T),
        )

    def set_temperature_and_pressure_from_altitude(
        self,
        h: float,
        units: str = "SI",
    ) -> None:
        """Sets the temperature, pressure and altitude of the environment.
        It uses the standard atmosphere model and approximations from:
        1) https://www.omnicalculator.com/physics/air-pressure-at-altitude
        2) https://www.omnicalculator.com/physics/altitude-temperature

        Args:
            h (float): Altitude in meters
            units (str, optional): Defaults to "SI". Can be "SI" or "Imperial"

        """
        if units == "Imperial":
            h *= 0.3048

        self._set_altitude(h)
        self._set_temperature_from_altitude(h)
        self._set_pressure_from_altitude_and_temperature(h, self.temperature)

    @property
    def air_dynamic_viscosity(self) -> float:
        mu: float = 1.458e-6 * (self.temperature**1.5) / (self.temperature + 110.4)
        return mu

    @property
    def air_kinematic_viscosity(self) -> float:
        return self.air_dynamic_viscosity / self.air_density

    @property
    def air_density(self) -> float:
        return self.pressure / (self.SPECIFIC_GAS_CONSTANT * self.temperature)

    def get_speed_of_sound(
        self,
        gamma: float,
        air_gas_constant: float,
        temperature: float,
    ) -> float:
        ret = float(np.sqrt(gamma * air_gas_constant * temperature))
        return ret

    # def __str__(self) -> str:
    #     return f"Environment: {self.name}"

    def __str__(self) -> str:
        string: str = f"Environment: {self.name}"
        string += f"\n\tGravity: {self.GRAVITY}"
        string += f"\n\tTemperature: {self.temperature}"
        string += f"\n\tPressure: {self.pressure}"
        string += f"\n\tDensity: {self.air_density}"
        string += f"\n\tKinematic Viscosity: {self.air_kinematic_viscosity}"
        return string


EARTH_ISA: Environment = Environment("Earth_Stardard_Atmosphere")
