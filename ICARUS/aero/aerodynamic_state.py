"""
Aerodynamic State Module

This module defines the AerodynamicState class that encapsulates simulation metrics
for aerodynamic analysis including airspeed, angles of attack and sideslip,
altitude, density, mach number, and angular rates.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ICARUS.core.types import FloatArray


class AerodynamicState:
    """
    Class representing the aerodynamic state of a vehicle encompassing
    simulation metrics such as airspeed, angle of attack, sideslip angle,
    altitude, air density, mach number, and angular rates.

    This class provides a comprehensive representation of the aerodynamic
    conditions needed for flight simulation and analysis.

    Attributes:
        airspeed (float): Airspeed in m/s
        alpha (float): Angle of attack in degrees
        altitude (float | None): Altitude in meters (None if not specified)
        beta (float): Sideslip angle in degrees
        density (float): Air density in kg/m³
        mach (float | None): Mach number (None if not specified)
        rate_P (float): Roll rate in rad/s
        rate_Q (float): Pitch rate in rad/s
        rate_R (float): Yaw rate in rad/s
    """

    def __init__(
        self,
        airspeed: float = 100.0,
        alpha: float = 2.0,
        altitude: float | None = None,
        beta: float = 0.0,
        density: float = 1.225,
        mach: float | None = None,
        rate_P: float = 0.0,
        rate_Q: float = 0.0,
        rate_R: float = 0.0,
    ) -> None:
        """
        Initialize an AerodynamicState instance.

        Args:
            airspeed: Airspeed in m/s (default: 100.0)
            alpha: Angle of attack in degrees (default: 2.0)
            altitude: Altitude in meters (default: None)
            beta: Sideslip angle in degrees (default: 0.0)
            density: Air density in kg/m³ (default: 1.225)
            mach: Mach number (default: None)
            rate_P: Roll rate in rad/s (default: 0.0)
            rate_Q: Pitch rate in rad/s (default: 0.0)
            rate_R: Yaw rate in rad/s (default: 0.0)
        """
        self.airspeed = airspeed
        self.alpha = alpha
        self.altitude = altitude
        self.beta = beta
        self.density = density
        self.mach = mach
        self.rate_P = rate_P
        self.rate_Q = rate_Q
        self.rate_R = rate_R

    @property
    def alpha_rad(self) -> float:
        """Return angle of attack in radians."""
        return np.deg2rad(self.alpha)

    @property
    def beta_rad(self) -> float:
        """Return sideslip angle in radians."""
        return np.deg2rad(self.beta)

    @property
    def angular_rates(self) -> FloatArray:
        """Return angular rates as a numpy array [P, Q, R] in rad/s."""
        return np.array([self.rate_P, self.rate_Q, self.rate_R])

    @property
    def velocity_components(self) -> FloatArray:
        """
        Calculate velocity components in body frame.

        Returns:
            FloatArray: [u, v, w] velocity components in m/s
        """
        u = self.airspeed * np.cos(self.alpha_rad) * np.cos(self.beta_rad)
        v = self.airspeed * np.cos(self.alpha_rad) * np.sin(self.beta_rad)
        w = self.airspeed * np.sin(self.alpha_rad) * np.cos(self.beta_rad)
        return np.array([u, v, w])

    @property
    def dynamic_pressure(self) -> float:
        """Calculate dynamic pressure in Pa."""
        return 0.5 * self.density * self.airspeed**2

    def __getstate__(self) -> dict[str, Any]:
        """
        Convert the aerodynamic state to a dictionary.

        Returns:
            dict: Dictionary representation of the aerodynamic state
        """
        return {
            "airspeed": self.airspeed,
            "alpha": self.alpha,
            "altitude": self.altitude,
            "beta": self.beta,
            "density": self.density,
            "mach": self.mach,
            "rate_P": self.rate_P,
            "rate_Q": self.rate_Q,
            "rate_R": self.rate_R,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Set the aerodynamic state from a dictionary.

        Args:
            state: Dictionary containing aerodynamic state data
        """
        return AerodynamicState.__init__(
            self,
            airspeed=state.get("airspeed", 100.0),
            alpha=state.get("alpha", 2.0),
            altitude=state.get("altitude"),
            beta=state.get("beta", 0.0),
            density=state.get("density", 1.225),
            mach=state.get("mach"),
            rate_P=state.get("rate_P", 0.0),
            rate_Q=state.get("rate_Q", 0.0),
            rate_R=state.get("rate_R", 0.0),
        )

    def copy(self) -> AerodynamicState:
        """
        Create a copy of the current aerodynamic state.

        Returns:
            AerodynamicState: Copy of the current instance
        """
        return AerodynamicState(
            airspeed=self.airspeed,
            alpha=self.alpha,
            altitude=self.altitude,
            beta=self.beta,
            density=self.density,
            mach=self.mach,
            rate_P=self.rate_P,
            rate_Q=self.rate_Q,
            rate_R=self.rate_R,
        )

    def update(self, **kwargs: Any) -> None:
        """
        Update aerodynamic state parameters.

        Args:
            **kwargs: Keyword arguments for parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"AerodynamicState has no attribute '{key}'")

    def __str__(self) -> str:
        """String representation of the aerodynamic state."""
        return (
            f"AerodynamicState(\n"
            f"  airspeed={self.airspeed:.2f} m/s\n"
            f"  alpha={self.alpha:.2f}°\n"
            f"  beta={self.beta:.2f}°\n"
            f"  altitude={self.altitude} m\n"
            f"  density={self.density:.3f} kg/m³\n"
            f"  mach={self.mach}\n"
            f"  rate_P={self.rate_P:.4f} rad/s\n"
            f"  rate_Q={self.rate_Q:.4f} rad/s\n"
            f"  rate_R={self.rate_R:.4f} rad/s\n"
            f")"
        )

    def __repr__(self) -> str:
        """Detailed representation of the aerodynamic state."""
        return (
            f"AerodynamicState("
            f"airspeed={self.airspeed}, "
            f"alpha={self.alpha}, "
            f"altitude={self.altitude}, "
            f"beta={self.beta}, "
            f"density={self.density}, "
            f"mach={self.mach}, "
            f"rate_P={self.rate_P}, "
            f"rate_Q={self.rate_Q}, "
            f"rate_R={self.rate_R})"
        )

    # def __eq__(self, other: object) -> bool:
    #     """Check equality with another AerodynamicState instance."""
    #     if not isinstance(other, AerodynamicState):
    #         return NotImplemented

    #     return (
    #         np.isclose(self.airspeed, other.airspeed)
    #         and np.isclose(self.alpha, other.alpha)
    #         and (
    #             (self.altitude is None and other.altitude is None)
    #             or (
    #                 self.altitude is not None
    #                 and other.altitude is not None
    #                 and np.isclose(self.altitude, other.altitude)
    #             )
    #         )
    #         and np.isclose(self.beta, other.beta)
    #         and np.isclose(self.density, other.density)
    #         and (
    #             (self.mach is None and other.mach is None)
    #             or (self.mach is not None and other.mach is not None and np.isclose(self.mach, other.mach))
    #         )
    #         and np.isclose(self.rate_P, other.rate_P)
    #         and np.isclose(self.rate_Q, other.rate_Q)
    #         and np.isclose(self.rate_R, other.rate_R)
    #     )
