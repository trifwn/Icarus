from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class AirfoilOperatingConditions:  # Or AirfoilOperatingPoint is also good
    """
    Represents the operational conditions under which an airfoil performs.

    Attributes:
        angle_of_attack_degrees (float): Angle of attack in degrees.
        reynolds_number (float): Reynolds number.
        mach_number (float): Mach number.
    """

    aoa: float
    reynolds_number: float
    mach_number: float


@dataclass
class AirfoilPressure:
    """
    Represents the pressure coefficient distribution along an airfoil's surface.

    Attributes:
        x (list[float]): X coordinates
        y (list[float]): Y coordinates
        cp (list[float]): Pressure coefficients (Cp) at the corresponding positions.
    """

    x: list[float] | npt.NDArray[np.float64]
    y: list[float] | npt.NDArray[np.float64]
    cp: list[float] | npt.NDArray[np.float64]


# @dataclass
# class AirfoilSurfacePressureDistribution:
#     """
#     Represents the pressure coefficient distribution on a single surface (upper or lower) of an airfoil.

#     Attributes:
#         x (list[float]): Normalized chord positions (x/c) where Cp values are recorded.
#         cp (list[float]): Pressure coefficients (Cp) at the corresponding chord positions.
#     """

#     x: list[float]
#     cp: list[float]


# @dataclass
# class AirfoilCompletePressureDistribution:
#     """
#     Encapsulates pressure coefficient distributions for both upper and lower surfaces of an airfoil.

#     Attributes:
#         upper_surface_distribution (AirfoilSurfacePressureDistribution): CP distribution for the upper surface.
#         lower_surface_distribution (AirfoilSurfacePressureDistribution): CP distribution for the lower surface.
#     """

#     upper_surface_distribution: AirfoilSurfacePressureDistribution
#     lower_surface_distribution: AirfoilSurfacePressureDistribution


@dataclass
class AirfoilOperatingPointMetrics:
    """
    Represents the comprehensive aerodynamic performance metrics of an airfoil
    at a specific set of operating conditions.

    Attributes:
        operating_conditions (AirfoilOperatingConditions): The conditions under which the results were obtained.
        Cl (float): The total lift coefficient (CL).
        Cd (float): The total drag coefficient (CD).
        Cp (float): The total pressure coefficient (CP), typically calculated as Cp = Cl / (0.5 * rho * V^2).
        CP_distribution (AirfoilPressureDistribution): Minimum pressure coefficient.
    """

    operating_conditions: AirfoilOperatingConditions
    Cl: float
    Cd: float
    Cm: float
    Cp_min: float
    Cp_distribution: AirfoilPressure | None = None
