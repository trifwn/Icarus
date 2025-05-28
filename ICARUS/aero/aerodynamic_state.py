"""
Aerodynamic State Module

This module defines the AerodynamicState class that encapsulates simulation metrics
for aerodynamic analysis including airspeed, angles of attack and sideslip,
altitude, density, mach number, and angular rates.
"""

from __future__ import annotations

from typing import Any

from jax import Array
import jax.numpy as jnp
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
        alpha: float = 0.0,
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
        self._airspeed = airspeed
        self._alpha = alpha
        self._altitude = altitude
        self._beta = beta
        self._density = density
        self._mach = mach
        self._rate_P = rate_P
        self._rate_Q = rate_Q
        self._rate_R = rate_R

    @property
    def airspeed(self) -> float:
        """Airspeed in m/s."""
        return self._airspeed

    @airspeed.setter
    def airspeed(self, value: float) -> None:
        """Set airspeed in m/s."""
        self._airspeed = value

    @property
    def alpha(self) -> float:
        """Angle of attack in degrees."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Set angle of attack in degrees."""
        self._alpha = value

    @property
    def altitude(self) -> float | None:
        """Altitude in meters (None if not specified)."""
        return self._altitude

    @altitude.setter
    def altitude(self, value: float | None) -> None:
        """Set altitude in meters."""
        self._altitude = value

    @property
    def beta(self) -> float:
        """Sideslip angle in degrees."""
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        """Set sideslip angle in degrees."""
        self._beta = value

    @property
    def density(self) -> float:
        """Air density in kg/m³."""
        return self._density

    @density.setter
    def density(self, value: float) -> None:
        """Set air density in kg/m³."""
        self._density = value

    @property
    def mach(self) -> float | None:
        """Mach number (None if not specified)."""
        return self._mach

    @mach.setter
    def mach(self, value: float | None) -> None:
        """Set Mach number."""
        self._mach = value

    @property
    def rate_P(self) -> float:
        """Roll rate in rad/s."""
        return self._rate_P

    @rate_P.setter
    def rate_P(self, value: float) -> None:
        """Set roll rate in rad/s."""
        self._rate_P = value

    @property
    def rate_Q(self) -> float:
        """Pitch rate in rad/s."""
        return self._rate_Q

    @rate_Q.setter
    def rate_Q(self, value: float) -> None:
        """Set pitch rate in rad/s."""
        self._rate_Q = value

    @property
    def rate_R(self) -> float:
        """Yaw rate in rad/s."""
        return self._rate_R

    @rate_R.setter
    def rate_R(self, value: float) -> None:
        """Set yaw rate in rad/s."""
        self._rate_R = value

    @property
    def alpha_rad(self) -> float:
        """Return angle of attack in radians."""
        return float(np.deg2rad(self.alpha))

    @property
    def beta_rad(self) -> float:
        """Return sideslip angle in radians."""
        return float(np.deg2rad(self.beta))

    @property
    def angular_rates(self) -> FloatArray:
        """Return angular rates as a numpy array [P, Q, R] in rad/s."""
        return np.array([self.rate_P, self.rate_Q, self.rate_R], dtype=float)

    @property
    def velocity_components(self) -> tuple[float, float, float]:
        """
        Calculate velocity components in body frame.

        Returns:
            FloatArray: [u, v, w] velocity components in m/s
        """
        u = self.airspeed * np.cos(self.alpha_rad) * np.cos(self.beta_rad)
        v = self.airspeed * np.cos(self.alpha_rad) * np.sin(self.beta_rad)
        w = self.airspeed * np.sin(self.alpha_rad) * np.cos(self.beta_rad)
        return u, v, w

    @property
    def velocity_vector(self) -> FloatArray:
        """
        Calculate the velocity vector in body frame.

        Returns:
            FloatArray: Velocity vector [u, v, w] in m/s
        """
        u, v, w = self.velocity_components
        return np.array([u, v, w], dtype=float)

    @property
    def velocity_vector_jax(self) -> Array:
        """
        Calculate the velocity vector in body frame using JAX.

        Returns:
            FloatArray: Velocity vector [u, v, w] in m/s
        """
        u, v, w = self.velocity_components
        return jnp.array([u, v, w])

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
