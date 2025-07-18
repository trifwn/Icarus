"""
Backward compatibility shims for deprecated methods and API changes.

This module provides compatibility shims to ensure that existing code continues
to work with the new JaxAirfoil implementation while providing deprecation warnings
and migration guidance.
"""

import functools
import warnings
from typing import Any
from typing import Callable
from typing import Optional

from .jax_implementation.jax_airfoil import JaxAirfoil


def deprecated(reason: str, replacement: Optional[str] = None):
    """
    Decorator to mark functions as deprecated.

    Args:
        reason: Reason for deprecation
        replacement: Suggested replacement function/method
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"{func.__name__} is deprecated: {reason}"
            if replacement:
                message += f" Use {replacement} instead."
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


class DeprecatedAirfoilMethods:
    """
    Container for deprecated methods that may still be used in legacy code.

    These methods provide backward compatibility while issuing deprecation warnings.
    """

    @staticmethod
    @deprecated(
        "repanel_spl is deprecated in favor of repanel with distribution='arc_length'",
        "airfoil.repanel(n_points, distribution='arc_length')",
    )
    def repanel_spl(airfoil: JaxAirfoil, n_points: int = 200) -> None:
        """
        Deprecated: Use repanel with arc_length distribution instead.

        This method modifies the airfoil in place, which is not compatible with
        JAX's functional programming model. Use repanel() which returns a new instance.
        """
        # For backward compatibility, we'll modify the airfoil in place
        # by replacing its internal data with the repaneled version
        repaneled = airfoil.repanel(n_points, distribution="arc_length")

        # Copy the repaneled data back to the original airfoil
        # Note: This breaks JAX's immutability but maintains API compatibility
        airfoil._coordinates = repaneled._coordinates
        airfoil._validity_mask = repaneled._validity_mask
        airfoil._n_valid_points = repaneled._n_valid_points
        airfoil._upper_split_idx = repaneled._upper_split_idx
        airfoil._max_buffer_size = repaneled._max_buffer_size

    @staticmethod
    @deprecated(
        "repanel_from_internal is deprecated in favor of repanel",
        "airfoil.repanel(n_points, distribution='cosine')",
    )
    def repanel_from_internal(
        airfoil: JaxAirfoil,
        n_points: int,
        distribution: str = "cosine",
        method: str = "interpolation",
    ) -> None:
        """
        Deprecated: Use repanel method instead.

        This method modifies the airfoil in place, which is not compatible with
        JAX's functional programming model.
        """
        # For backward compatibility, modify in place
        repaneled = airfoil.repanel(n_points, distribution=distribution)

        # Copy the repaneled data back to the original airfoil
        airfoil._coordinates = repaneled._coordinates
        airfoil._validity_mask = repaneled._validity_mask
        airfoil._n_valid_points = repaneled._n_valid_points
        airfoil._upper_split_idx = repaneled._upper_split_idx
        airfoil._max_buffer_size = repaneled._max_buffer_size

    @staticmethod
    @deprecated(
        "Direct coordinate modification is deprecated",
        "Create a new JaxAirfoil instance with modified coordinates",
    )
    def modify_coordinates_in_place(airfoil: JaxAirfoil, new_coordinates: Any) -> None:
        """
        Deprecated: Direct coordinate modification breaks JAX immutability.

        Instead of modifying coordinates in place, create a new JaxAirfoil instance.
        """
        warnings.warn(
            "Modifying coordinates in place is not recommended with JaxAirfoil. "
            "Consider creating a new instance with JaxAirfoil(coordinates=new_coordinates)",
            UserWarning,
        )


class LegacyAPIShim:
    """
    Provides shims for legacy API patterns that don't directly translate to JAX.

    This class helps bridge the gap between the original Airfoil API and the new
    JAX-compatible implementation.
    """

    @staticmethod
    def ensure_jax_compatibility(method_name: str, *args, **kwargs):
        """
        Ensure that method calls are compatible with JAX transformations.

        Args:
            method_name: Name of the method being called
            *args: Method arguments
            **kwargs: Method keyword arguments
        """
        # Check for common incompatible patterns
        if "inplace" in kwargs and kwargs["inplace"]:
            warnings.warn(
                f"{method_name} with inplace=True is not compatible with JAX. "
                "The method will return a new instance instead.",
                UserWarning,
            )
            kwargs["inplace"] = False

        return args, kwargs

    @staticmethod
    def handle_numpy_inputs(inputs: Any) -> Any:
        """
        Convert NumPy inputs to JAX arrays with appropriate warnings.

        Args:
            inputs: Input data that might be NumPy arrays

        Returns:
            JAX-compatible inputs
        """
        import jax.numpy as jnp
        import numpy as np

        if isinstance(inputs, np.ndarray):
            warnings.warn(
                "NumPy arrays are automatically converted to JAX arrays. "
                "For better performance, consider using JAX arrays directly.",
                UserWarning,
            )
            return jnp.array(inputs)
        elif isinstance(inputs, (list, tuple)):
            # Recursively handle nested structures
            return type(inputs)(
                LegacyAPIShim.handle_numpy_inputs(item) for item in inputs
            )

        return inputs

    @staticmethod
    def wrap_method_for_compatibility(method: Callable) -> Callable:
        """
        Wrap a method to provide legacy API compatibility.

        Args:
            method: Method to wrap

        Returns:
            Wrapped method with compatibility features
        """

        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            # Handle NumPy inputs
            args = tuple(LegacyAPIShim.handle_numpy_inputs(arg) for arg in args)
            kwargs = {
                k: LegacyAPIShim.handle_numpy_inputs(v) for k, v in kwargs.items()
            }

            # Ensure JAX compatibility
            args, kwargs = LegacyAPIShim.ensure_jax_compatibility(
                method.__name__,
                *args,
                **kwargs,
            )

            return method(*args, **kwargs)

        return wrapper


def add_compatibility_methods(jax_airfoil_class):
    """
    Add compatibility methods to the JaxAirfoil class.

    This function dynamically adds deprecated methods to maintain backward compatibility.

    Args:
        jax_airfoil_class: The JaxAirfoil class to modify
    """

    # Add deprecated methods as instance methods
    def repanel_spl_method(self, n_points: int = 200) -> None:
        return DeprecatedAirfoilMethods.repanel_spl(self, n_points)

    def repanel_from_internal_method(
        self,
        n_points: int,
        distribution: str = "cosine",
        method: str = "interpolation",
    ) -> None:
        return DeprecatedAirfoilMethods.repanel_from_internal(
            self,
            n_points,
            distribution,
            method,
        )

    # Add methods to the class if they don't already exist
    if not hasattr(jax_airfoil_class, "repanel_spl"):
        jax_airfoil_class.repanel_spl = repanel_spl_method

    if not hasattr(jax_airfoil_class, "repanel_from_internal"):
        jax_airfoil_class.repanel_from_internal = repanel_from_internal_method


# Apply compatibility methods to JaxAirfoil
add_compatibility_methods(JaxAirfoil)


class APICompatibilityChecker:
    """
    Utility to check API compatibility between old and new implementations.
    """

    @staticmethod
    def check_method_compatibility(old_class, new_class) -> dict:
        """
        Check which methods are available in both classes.

        Args:
            old_class: Original Airfoil class
            new_class: New JaxAirfoil class

        Returns:
            Dictionary with compatibility information
        """
        old_methods = set(dir(old_class))
        new_methods = set(dir(new_class))

        return {
            "common_methods": old_methods & new_methods,
            "missing_in_new": old_methods - new_methods,
            "new_methods": new_methods - old_methods,
            "compatibility_score": len(old_methods & new_methods) / len(old_methods),
        }

    @staticmethod
    def suggest_migration_path(method_name: str) -> str:
        """
        Suggest migration path for specific methods.

        Args:
            method_name: Name of the method to migrate

        Returns:
            Migration suggestion
        """
        migration_map = {
            "repanel_spl": "Use airfoil.repanel(n_points, distribution='arc_length')",
            "repanel_from_internal": "Use airfoil.repanel(n_points, distribution='cosine')",
            "modify_coordinates": "Create new JaxAirfoil instance with new coordinates",
            "in_place_operations": "Use functional style - methods return new instances",
        }

        return migration_map.get(
            method_name,
            "Check documentation for JAX-compatible alternative",
        )
