"""
Migration utilities for converting between Airfoil and JaxAirfoil instances.

This module provides utilities to help users migrate from the original Airfoil class
to the new JaxAirfoil class, and vice versa, while maintaining data integrity and
API compatibility.
"""

import warnings
from typing import Any
from typing import Dict
from typing import Optional

import jax.numpy as jnp
import numpy as np

from .airfoil import Airfoil
from .core.airfoil_geometry import AirfoilGeometry


class AirfoilMigrationUtils:
    """Utilities for migrating between Airfoil and JaxAirfoil instances."""

    @staticmethod
    def airfoil_to_airfoil_geometry(
        airfoil: Airfoil,
        buffer_size: Optional[int] = None,
        preserve_metadata: bool = True,
    ) -> AirfoilGeometry:
        """
        Convert an existing Airfoil instance to a JaxAirfoil instance.

        Args:
            airfoil: Original Airfoil instance to convert
            buffer_size: Optional buffer size for the JaxAirfoil (auto-determined if None)
            preserve_metadata: Whether to preserve metadata from the original airfoil

        Returns:
            JaxAirfoil instance with equivalent geometry and properties

        Example:
            >>> original_airfoil = Airfoil.naca("2412", n_points=100)
            >>> airfoil_geometry = AirfoilMigrationUtils.airfoil_to_airfoil_geometry(original_airfoil)
            >>> # Now you can use JAX features like JIT compilation and gradients
            >>> airfoil_geometry = jax.jit(lambda af: af.thickness(0.5))(airfoil_geometry)
        """
        # Get coordinates from the original airfoil
        try:
            # Try to get upper and lower surface coordinates
            x_upper, y_upper = airfoil.upper_surface_points
            x_lower, y_lower = airfoil.lower_surface_points

            # Stack into coordinate arrays
            upper_coords = jnp.stack([x_upper, y_upper])
            lower_coords = jnp.stack([x_lower, y_lower])

        except AttributeError:
            # Fallback: get all coordinates and split them
            try:
                x_coords, y_coords = airfoil.get_coordinates()
                coords = jnp.stack([x_coords, y_coords])

                # Create JaxAirfoil from selig format coordinates
                metadata = (
                    {"name": getattr(airfoil, "name", "Migrated_Airfoil")}
                    if preserve_metadata
                    else None
                )
                return AirfoilGeometry(
                    coordinates=coords,
                    name=getattr(airfoil, "name", "Migrated_Airfoil"),
                    buffer_size=buffer_size,
                    metadata=metadata,
                )
            except AttributeError:
                raise ValueError(
                    "Cannot extract coordinates from the provided Airfoil instance. "
                    "The airfoil must have either 'upper_surface_points' and 'lower_surface_points' "
                    "properties or a 'get_coordinates()' method.",
                )

        # Prepare metadata
        metadata = {}
        if preserve_metadata:
            # Extract common metadata attributes
            for attr in ["name", "naca", "file_name"]:
                if hasattr(airfoil, attr):
                    value = getattr(airfoil, attr)
                    if value is not None:
                        metadata[attr] = value

        # Create JaxAirfoil from upper and lower surfaces
        return AirfoilGeometry.from_upper_lower(
            upper_coords,
            lower_coords,
            name=metadata.get("name", "Migrated_Airfoil"),
            buffer_size=buffer_size,
            metadata=metadata,
        )

    @staticmethod
    def airfoil_geometry_to_airfoil(airfoil_geometry: AirfoilGeometry) -> Airfoil:
        """
        Convert a JaxAirfoil instance to an original Airfoil instance.

        Args:
            airfoil_geometry: JaxAirfoil instance to convert

        Returns:
            Airfoil instance with equivalent geometry

        Note:
            This conversion will lose JAX-specific features like automatic differentiation
            and JIT compilation capabilities.

        Example:
            >>> airfoil_geometry = JaxAirfoil.naca("2412", n_points=100)
            >>> original_airfoil = AirfoilMigrationUtils.airfoil_geometry_to_airfoil(airfoil_geometry)
        """
        # Get upper and lower surface coordinates
        x_upper, y_upper = airfoil_geometry.upper_surface_points
        x_lower, y_lower = airfoil_geometry.lower_surface_points

        # Convert JAX arrays to NumPy arrays
        upper_coords = np.stack([np.array(x_upper), np.array(y_upper)])
        lower_coords = np.stack([np.array(x_lower), np.array(y_lower)])

        # Create original Airfoil instance
        return Airfoil(
            upper=upper_coords,
            lower=lower_coords,
            name=airfoil_geometry.name,
        )

    @staticmethod
    def batch_migrate_to_jax(
        airfoils: list[Airfoil],
        buffer_size: Optional[int] = None,
        preserve_metadata: bool = True,
    ) -> list[AirfoilGeometry]:
        """
        Convert a batch of Airfoil instances to JaxAirfoil instances.

        Args:
            airfoils: List of Airfoil instances to convert
            buffer_size: Optional buffer size for all JaxAirfoils (auto-determined if None)
            preserve_metadata: Whether to preserve metadata from original airfoils

        Returns:
            List of JaxAirfoil instances

        Example:
            >>> airfoils = [Airfoil.naca(f"00{i:02d}", n_points=100) for i in range(8, 16, 2)]
            >>> airfoil_geometrys = AirfoilMigrationUtils.batch_migrate_to_jax(airfoils)
        """
        airfoil_geometrys = []

        for airfoil in airfoils:
            try:
                airfoil_geometry = AirfoilMigrationUtils.airfoil_to_airfoil_geometry(
                    airfoil,
                    buffer_size=buffer_size,
                    preserve_metadata=preserve_metadata,
                )
                airfoil_geometrys.append(airfoil_geometry)
            except Exception as e:
                warnings.warn(
                    f"Failed to convert airfoil {getattr(airfoil, 'name', 'Unknown')}: {e}",
                    UserWarning,
                )
                continue

        return airfoil_geometrys

    @staticmethod
    def validate_migration(
        original: Airfoil,
        migrated: AirfoilGeometry,
        tolerance: float = 1e-10,
    ) -> Dict[str, bool]:
        """
        Validate that a migration preserved the airfoil geometry correctly.

        Args:
            original: Original Airfoil instance
            migrated: Migrated JaxAirfoil instance
            tolerance: Numerical tolerance for comparison

        Returns:
            Dictionary with validation results for different properties

        Example:
            >>> original = Airfoil.naca("2412", n_points=100)
            >>> migrated = AirfoilMigrationUtils.airfoil_to_airfoil_geometry(original)
            >>> results = AirfoilMigrationUtils.validate_migration(original, migrated)
            >>> print("Migration successful:", all(results.values()))
        """
        results = {}

        try:
            # Compare coordinates
            x_orig_upper, y_orig_upper = original.upper_surface_points
            x_orig_lower, y_orig_lower = original.lower_surface_points

            x_migr_upper, y_migr_upper = migrated.upper_surface_points
            x_migr_lower, y_migr_lower = migrated.lower_surface_points

            # Check upper surface
            results["upper_surface_x"] = np.allclose(
                x_orig_upper,
                x_migr_upper,
                atol=tolerance,
            )
            results["upper_surface_y"] = np.allclose(
                y_orig_upper,
                y_migr_upper,
                atol=tolerance,
            )

            # Check lower surface
            results["lower_surface_x"] = np.allclose(
                x_orig_lower,
                x_migr_lower,
                atol=tolerance,
            )
            results["lower_surface_y"] = np.allclose(
                y_orig_lower,
                y_migr_lower,
                atol=tolerance,
            )

        except Exception as e:
            warnings.warn(f"Could not validate coordinates: {e}", UserWarning)
            results["coordinates"] = False

        try:
            # Compare geometric properties if available
            if hasattr(original, "max_thickness") and hasattr(
                migrated,
                "max_thickness",
            ):
                results["max_thickness"] = (
                    abs(original.max_thickness - migrated.max_thickness) < tolerance
                )

            if hasattr(original, "max_camber") and hasattr(migrated, "max_camber"):
                results["max_camber"] = (
                    abs(original.max_camber - migrated.max_camber) < tolerance
                )

        except Exception as e:
            warnings.warn(f"Could not validate geometric properties: {e}", UserWarning)

        return results

    @staticmethod
    def create_compatibility_wrapper(
        airfoil_geometry: AirfoilGeometry,
    ) -> "CompatibilityWrapper":
        """
        Create a compatibility wrapper that provides the original Airfoil API.

        Args:
            airfoil_geometry: JaxAirfoil instance to wrap

        Returns:
            CompatibilityWrapper instance that behaves like the original Airfoil

        Example:
            >>> airfoil_geometry = JaxAirfoil.naca("2412", n_points=100)
            >>> wrapper = AirfoilMigrationUtils.create_compatibility_wrapper(airfoil_geometry)
            >>> # Use wrapper with existing code that expects original Airfoil API
        """
        return CompatibilityWrapper(airfoil_geometry)


class CompatibilityWrapper:
    """
    Wrapper class that provides original Airfoil API while using JaxAirfoil internally.

    This wrapper allows existing code to work with JaxAirfoil instances without modification,
    while still providing access to JAX features when needed.
    """

    def __init__(self, airfoil_geometry: AirfoilGeometry):
        """Initialize the compatibility wrapper."""
        self._airfoil_geometry = airfoil_geometry

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying JaxAirfoil."""
        return getattr(self._airfoil_geometry, name)

    @property
    def airfoil_geometry(self) -> AirfoilGeometry:
        """Access the underlying JaxAirfoil for JAX-specific operations."""
        return self._airfoil_geometry

    # Provide explicit implementations for commonly used methods to ensure compatibility
    @property
    def upper_surface_points(self):
        """Get upper surface points."""
        return self._airfoil_geometry.upper_surface_points

    @property
    def lower_surface_points(self):
        """Get lower surface points."""
        return self._airfoil_geometry.lower_surface_points

    def get_coordinates(self):
        """Get all coordinates."""
        return self._airfoil_geometry.get_coordinates()

    def thickness(self, x):
        """Compute thickness at given x-coordinates."""
        return self._airfoil_geometry.thickness(x)

    def camber_line(self, x):
        """Compute camber line at given x-coordinates."""
        return self._airfoil_geometry.camber_line(x)

    def y_upper(self, x):
        """Query upper surface y-coordinates."""
        return self._airfoil_geometry.y_upper(x)

    def y_lower(self, x):
        """Query lower surface y-coordinates."""
        return self._airfoil_geometry.y_lower(x)

    def plot(self, *args, **kwargs):
        """Plot the airfoil."""
        return self._airfoil_geometry.plot(*args, **kwargs)

    @property
    def name(self):
        """Get airfoil name."""
        return self._airfoil_geometry.name

    @property
    def max_thickness(self):
        """Get maximum thickness."""
        return self._airfoil_geometry.max_thickness

    @property
    def max_camber(self):
        """Get maximum camber."""
        return self._airfoil_geometry.max_camber
