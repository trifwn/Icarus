import numpy as np

class Environment():
    def __init__(self):
        self.Gravity = 9.81
        self.AirDensity = 1.225
        self.AirDynamicViscosity = 1.7894e-5
        self.AirKinematicViscosity = 1.7894e-5
        
        self.Temperature = 20 + 273.15 
        self.Gamma = 1.4
        self.R = 287.058
        
        # Speed of sound
        self.AirSpeedOfSound = self.getMachSpeed(self.Gamma, self.R, self.Temperature)
        
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
        
    def getMachSpeed(self,gamma, AirGasConstant, temperature):
        return np.sqrt(gamma * AirGasConstant * temperature)
    
