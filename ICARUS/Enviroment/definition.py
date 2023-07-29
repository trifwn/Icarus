import numpy as np


class Environment:
    def __init__(self, name: str) -> None:
        """
        Definition of the Environment object

        Args:
            name (str): Name of the environment
        """
        self.name: str = name
        self.GRAVITY: float = 9.81
        self.air_density: float = 1.225  # self.get_density_from_altitude(altitude) #1.225
        self.air_dynamic_viscosity: float = 1.56e-5
        self.air_kinematic_viscosity: float = 1.56e-5
        self.air_temperature: float = 20 + 273.15

        self.gamma: float = 1.4
        self.R: float = 287.058

        # Speed of sound
        self.air_speed_of_sound: float = self.get_speed_of_sound(
            self.gamma,
            self.R,
            self.air_temperature,
        )

        # Chemical properties
        self.air_molar_mass: float = 0.0289644
        self.GAS_CONSTANT: float = 8.31447

        # Thermodynamic properties
        self.air_specific_heat_constant_pressure: float = 1005.7
        self.air_specific_heat_constant_volume: float = 718.7

        # Thermal properties
        self.air_thermal_conductivity: float = 0.0257
        self.air_thermal_diffusivity: float = 1.25e-5
        self.air_thermal_expansion_coefficient: float = 0.000012
        self.air_prandtl_number: float = 0.71

    # def get_density_from_altitude(self, altitude: float)-> float
    #     """
    #     Get Density
    #     TODO: FIND RELATION

    #     Returns:
    #         _type_: _description_
    #     """

    #     return 1.225

    def get_speed_of_sound(
        self,
        gamma: float,
        air_gas_constant: float,
        temperature: float,
    ) -> float:
        ret = float(np.sqrt(gamma * air_gas_constant * temperature))
        return ret

    def __str__(self) -> str:
        return f"Environment: {self.name}"

    # def __str__(self):
    #     str = f"Environment: {self.name}"
    #     str += f"\n\tGravity: {self.Gravity}"
    #     str += f"\n\tDensity: {self.AirDensity}"
    #     str += f"\n\tViscosity: {self.AirKinematicViscosity}"
    #     str += f"\n\tTemperature: {self.Temperature}"
    #     return str


EARTH = Environment("Earth")
