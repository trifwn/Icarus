"""
Defines functions usefull for the cruising state of the flight and
the cruise class
"""
import numpy as np
from numpy import dtype, floating, ndarray
from typing import Any

from .plane_concept import ConceptualPlane
from .wing_concept import ConceptualWing


class Cruise():
    "Cruise Class"

    def __init__(
        self,
        plane: ConceptualPlane,
        altitude: float,
        velocity: float
    ) -> None:
        self.plane: ConceptualPlane = plane
        self.altitude = altitude
        self.cruise_velocity = velocity
        self.S: float = self.plane.S
        self.W: float = self.plane.W

        self._set_air_density(altitude)

    def _set_air_density(self, altitude: float) -> None:
        """
        Returns the air density of the cruise
        Based on https://www.weather.gov/media/epz/wxcalc/densityAltitude.pdf
        """
        self.air_density = self.sea_level_air_density

    def _set_velocities(self, angle: float) -> None:
        """Returns the velocity of the cruise

        Args:
            angle (float): Angle where the plane is trimmed
        """
        cl: float = float(np.interp(angle, self.plane.main_wing.aoa, self.plane.cl))
        self.cruise_velocity = (
            self.W /
            (0.5 * self.air_density * self.S * cl)
            ) ** 0.5

        self.min_velocity = (
            self.W /
            (0.5 * self.air_density * self.S * self.plane.cl_max)
            ) ** 0.5

        self.apparent_velocity = np.sqrt(
            self.air_density / self.sea_level_air_density
            ) * self.cruise_velocity

    def _set_dynamic_pressure(self) -> None:
        """Returns the dynamic pressure of the cruise"""
        self.dyn_pressure: float = (
            0.5 * self.air_density * self.cruise_velocity
            ) ** 2

    @property
    def sigma(self) -> float:
        """Returns the density ratio"""
        return self.air_density / self.sea_level_air_density

    def _set_cl(self) -> None:
        """Returns the lift coefficient of the cruise based on the
        apparent velocity
        """
        self.cl: float = (
            self.W /
            (0.5 * self.sea_level_air_density * self.S) *
            self.apparent_velocity ** (-2)
        )

    def surface_lift(self, wing: ConceptualWing) -> ndarray[Any, dtype[floating]]:
        """Returns the surface lift of a lifting surface"""
        return wing.cl_3d * self.dyn_pressure * wing.area

    def surface_drag(self, wing: ConceptualWing, k: float) -> float:
        """Returns the drag of the plane
        Drag is calculated according to:
        Drag = C_D * q * S
        -> C_D = C_D_0 + k * C_L^2
        => Drag = k1 V^2 + k2 / (V^2)
        """
        cd_0: float = wing.cd_0
        area: float = wing.area
        k_1: float = cd_0 * self.air_density / (2 * area)
        k_2: float = (
            k * self.W /
            (0.5 * self.air_density * area)
        )

        drag: float = (
            k_1*self.cruise_velocity**2 +
            k_2 / (self.cruise_velocity**2)
        )
        return drag

    @property
    def k(self) -> float:
        e: float = self.plane.main_wing.oswald_efficiency
        return 1 / (
            np.pi * self.plane.main_wing.aspect_ratio * e
            )
    
    def plane_drag(self, wings: list[ConceptualWing], k: float) -> float:
        """Returns the least drag of the plane"""
        drag: float = 0
        for wing in wings:
            drag += self.surface_drag(wing, k)
        return drag

    def least_drag_conditions(
        self,
        plane: ConceptualPlane,
    ) -> tuple[float, float, float]:
        """Returns the least drag of the plane

        Args:
            plane (ConceptualPlane): Plane Object

        Returns:
            tuple[float, float, float]: Minimum Drag,
                                        Velocity at min Drag,
                                        Effective Velocity.
        """
        k: float = self.k
        wing: ConceptualWing = plane.main_wing
        drag_min: float = 2 * self.W * np.sqrt(k * wing.cd_0)
        u_at_drag_min: float = np.sqrt(
                self.W /
                (0.5 * self.air_density * wing.area)
            )
        u_eff_at_drag_min: float = u_at_drag_min * np.sqrt(self.sigma)
        # c_l_least_drag: float = np.sqrt(wing.cd_0 / k)
        # c_d_least_drag: float = 2 * wing.cd_0
        # d_over_l_least_drag: float = 2 * np.sqrt(k * wing.cd_0)
        
        return drag_min, u_at_drag_min, u_eff_at_drag_min

    def power_consumption(self, velocity) -> float:
        """
        Returns the power consumption of the plane

        Args:
            velocity (float): Velocity of the plane

        Returns:
            float: Power consumption
        """
        P: float = (
            0.5 * self.plane.cd_0 * self.air_density * self.S * velocity**3 +
            self.k * self.W**2 / (0.5 * self.air_density * self.S) / velocity
        )
        return P
    
    def relative_power_consumption(self, velocity: float) -> float:
        _ , velocity_p_min , _ = self.least_power_conditions()
        p_min: float = self.power_consumption(velocity_p_min)
        return self.power_consumption(velocity) / p_min
        # IF U = n * u_at_power_min
        
    # def ascend(self, gamma: float):
    #     lift: float = self.W * np.cos(gamma)
    #     thrust: float = drag + self.W * np.sin(gamma)
    #     power: float = thrust * velocity
    #     gamma = np.arcsin (thrust-drag)/self.W
    #     d_h_d_t = (thrust * velocity - drag * velocity)/ self.W
    #     pass

    def least_power_conditions(self) -> tuple[float, float, float]:
        u_at_power_min: float = (
            (2 * self.W / self.air_density / self.S ) ** 0.5 + 
            (self.k / 3 / self.plane.cd_0) ** 0.25
        ) # 0.76 * V_least_drag

        u_eff_at_power_min: float =  (
            (2 * self.W / self.sea_level_air_density / self.S ) ** 0.5 + 
            (self.k / 3 / self.plane.cd_0) ** 0.25
        ) 
        
        # cl_power_min: float = np.sqrt(3 * self.plane.cd_0 ) / self.k
        drag_power_min: float = 4 * self.plane.cd_0 # 1.15 * Drag_least_drag
        # velocity_power_min
        return drag_power_min, u_at_power_min, u_eff_at_power_min

    def _get_elevator_lever(self, angle: float) -> float:
        """Returns the elevator lever arm"""
        elevator_lifts = self.surface_lift(self.plane.elevator)
        elevator_lift: float = float(np.interp(angle, self.plane.elevator.aoa, elevator_lifts))
        l_t: float = (
           self.W *
           (self.plane.cog - self.plane.main_wing.cp_pos) -
           elevator_lift *
           (1 + (self.plane.cog - self.plane.elevator.cp_pos))
           )

        return l_t
