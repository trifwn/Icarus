import numpy as np


class Environment:
    def __init__(self, name):
        self.name = name
        self.Gravity = 9.81
        self.AirDensity = 1.225
        self.AirDynamicViscosity = 1.56e-5
        self.AirKinematicViscosity = 1.56e-5

        self.Temperature = 20 + 273.15
        self.gamma = 1.4
        self.R = 287.058

        # Speed of sound
        self.AirSpeedOfSound = self.getMachSpeed(self.gamma, self.R, self.Temperature)

        # Chemical properties
        self.AirMolarMass = 0.0289644
        self.AirGasConstant = 8.31447

        # Thermodynamic properties
        self.AirSpecificHeatConstantPressure = 1005.7
        self.AirSpecificHeatConstantVolume = 718.7

        # Thermal properties
        self.AirThermalConductivity = 0.0257
        self.AirThermalDiffusivity = 1.25e-5
        self.AirThermalExpansionCoefficient = 0.000012
        self.AirThermalConductivity = 0.0257
        self.AirThermalDiffusivity = 1.25e-5
        self.AirPrandtlNumber = 0.71

    def getMachSpeed(self, gamma, AirGasConstant, temperature):
        return np.sqrt(gamma * AirGasConstant * temperature)

    def __str__(self):
        return f"Environment: {self.name}"

    # def __str__(self):
    #     str = f"Environment: {self.name}"
    #     str += f"\n\tGravity: {self.Gravity}"
    #     str += f"\n\tDensity: {self.AirDensity}"
    #     str += f"\n\tViscosity: {self.AirKinematicViscosity}"
    #     str += f"\n\tTemperature: {self.Temperature}"
    #     return str


EARTH = Environment("Earth")
