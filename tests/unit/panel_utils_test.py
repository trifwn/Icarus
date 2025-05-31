"""
Validation script for aerodynamic utility functions.

This script tests the JAX utility functions against precomputed data
from LSPT_Plane objects to ensure correctness and consistency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import vmap

if TYPE_CHECKING:
    from ICARUS.aero import LSPT_Plane
    from ICARUS.vehicle import Airplane

from ICARUS.aero.utils import panel_area
from ICARUS.aero.utils import panel_center
from ICARUS.aero.utils import panel_cp
from ICARUS.aero.utils import panel_cp_normal


class TestUtilityFunctions:
    """Test suite for aerodynamic utility functions."""

    def test_panel_cp_validation(self, test_plane: LSPT_Plane) -> None:
        """
        Test panel_cp function against LSPT_Plane precomputed control points.

        Args:
            test_plane: LSPT_Plane object with precomputed data
        """
        print("Testing panel_cp function...")

        # Get all surface panels (excluding wake panels)
        surface_panels = test_plane.panels[test_plane.panel_indices]
        precomputed_cps = test_plane.panel_cps[test_plane.panel_indices]

        # Test single panel
        test_panel = surface_panels[0]
        computed_cp = panel_cp(test_panel)
        expected_cp = precomputed_cps[0]

        print("Single panel test:")
        print(f"  Computed CP: {computed_cp}")
        print(f"  Expected CP: {expected_cp}")
        print(f"  Difference: {jnp.linalg.norm(computed_cp - expected_cp)}")

        # Tolerance for floating point comparison
        tolerance = 1e-6
        assert jnp.allclose(
            computed_cp,
            expected_cp,
            atol=tolerance,
        ), f"Single panel CP mismatch: {jnp.linalg.norm(computed_cp - expected_cp)}"

        # Test vectorized computation for all panels
        vectorized_panel_cp = vmap(panel_cp)
        computed_cps = vectorized_panel_cp(surface_panels)

        # Compare all control points
        max_error = jnp.max(jnp.linalg.norm(computed_cps - precomputed_cps, axis=1))
        mean_error = jnp.mean(jnp.linalg.norm(computed_cps - precomputed_cps, axis=1))

        print(f"Vectorized test ({len(surface_panels)} panels):")
        print(f"  Max error: {max_error}")
        print(f"  Mean error: {mean_error}")
        print(f"  All within tolerance: {jnp.allclose(computed_cps, precomputed_cps, atol=tolerance)}")

        assert jnp.allclose(
            computed_cps,
            precomputed_cps,
            atol=tolerance,
        ), f"Vectorized CP computation failed with max error: {max_error}"

        print("✓ panel_cp validation passed\n")

    def test_panel_control_nj_validation(self, test_plane: LSPT_Plane) -> None:
        """
        Test panel_control_nj function against LSPT_Plane precomputed normals.

        Args:
            test_plane: LSPT_Plane object with precomputed data
        """
        print("Testing panel_control_nj function...")

        # Get all surface panels and normals (excluding wake panels)
        surface_panels = test_plane.panels[test_plane.panel_indices]
        precomputed_normals = test_plane.panel_normals[test_plane.panel_indices]

        # Test single panel
        test_panel = surface_panels[0]
        computed_normal = panel_cp_normal(test_panel)
        expected_normal = precomputed_normals[0]

        print("Single panel test:")
        print(f"  Computed normal: {computed_normal}")
        print(f"  Expected normal: {expected_normal}")
        print(f"  Difference: {jnp.linalg.norm(computed_normal - expected_normal)}")
        print(f"  Computed magnitude: {jnp.linalg.norm(computed_normal)}")
        print(f"  Expected magnitude: {jnp.linalg.norm(expected_normal)}")

        # Check that both are unit vectors
        assert jnp.allclose(jnp.linalg.norm(computed_normal), 1.0, atol=1e-6), "Computed normal is not a unit vector"
        assert jnp.allclose(jnp.linalg.norm(expected_normal), 1.0, atol=1e-6), "Expected normal is not a unit vector"

        # Tolerance for floating point comparison
        tolerance = 1e-6
        assert jnp.allclose(
            computed_normal,
            expected_normal,
            atol=tolerance,
        ), f"Single panel normal mismatch: {jnp.linalg.norm(computed_normal - expected_normal)}"

        # Test vectorized computation for all panels
        vectorized_panel_nj = vmap(panel_cp_normal)
        computed_normals = vectorized_panel_nj(surface_panels)

        # Verify all computed normals are unit vectors
        computed_magnitudes = jnp.linalg.norm(computed_normals, axis=1)
        assert jnp.allclose(computed_magnitudes, 1.0, atol=1e-6), "Not all computed normals are unit vectors"

        # Compare all normals
        max_error = jnp.max(jnp.linalg.norm(computed_normals - precomputed_normals, axis=1))
        mean_error = jnp.mean(jnp.linalg.norm(computed_normals - precomputed_normals, axis=1))

        print(f"Vectorized test ({len(surface_panels)} panels):")
        print(f"  Max error: {max_error}")
        print(f"  Mean error: {mean_error}")
        print(f"  All within tolerance: {jnp.allclose(computed_normals, precomputed_normals, atol=tolerance)}")

        assert jnp.allclose(
            computed_normals,
            precomputed_normals,
            atol=tolerance,
        ), f"Vectorized normal computation failed with max error: {max_error}"

        print("✓ panel_control_nj validation passed\n")

    def test_panel_area_validation(self, test_plane: LSPT_Plane) -> None:
        """
        Test panel_area function by comparing with manual calculation.

        Args:
            test_plane: LSPT_Plane object with precomputed data
        """
        print("Testing panel_area function...")

        # Get surface panels
        surface_panels = test_plane.panels[test_plane.panel_indices]

        # Test single panel
        test_panel = surface_panels[0]
        computed_area = panel_area(test_panel)

        # Manual calculation for comparison
        # Area = 0.5 * |cross_product of diagonals|
        diagonal1 = test_panel[0] - test_panel[2]
        diagonal2 = test_panel[1] - test_panel[3]
        manual_area = 0.5 * jnp.linalg.norm(jnp.cross(diagonal1, diagonal2))

        print("Single panel test:")
        print(f"  Computed area: {computed_area}")
        print(f"  Manual area: {manual_area}")
        print(f"  Difference: {abs(computed_area - manual_area)}")

        tolerance = 1e-10
        assert jnp.allclose(
            computed_area,
            manual_area,
            atol=tolerance,
        ), f"Single panel area mismatch: {abs(computed_area - manual_area)}"

        # Test vectorized computation
        vectorized_panel_area = vmap(panel_area)
        computed_areas = vectorized_panel_area(surface_panels)

        # Verify all areas are positive
        assert jnp.all(computed_areas > 0), "Some computed areas are not positive"

        # Calculate total surface area
        total_area = jnp.sum(computed_areas)
        print(f"Vectorized test ({len(surface_panels)} panels):")
        print(f"  Total computed area: {total_area}")
        print(f"  Mean panel area: {jnp.mean(computed_areas)}")
        print(f"  Min panel area: {jnp.min(computed_areas)}")
        print(f"  Max panel area: {jnp.max(computed_areas)}")

        # Compare with reference area if available
        if hasattr(test_plane, "S"):
            print(f"  Reference area (S): {test_plane.S}")
            # Note: Total panel area might differ from reference area due to discretization

        print("✓ panel_area validation passed\n")

    def test_panel_center_validation(self, test_plane: LSPT_Plane) -> None:
        """
        Test panel_center function by comparing with manual calculation.

        Args:
            test_plane: LSPT_Plane object with precomputed data
        """
        print("Testing panel_center function...")

        # Get surface panels
        surface_panels = test_plane.panels[test_plane.panel_indices]

        # Test single panel
        test_panel = surface_panels[0]
        computed_center = panel_center(test_panel)

        # Manual calculation - simple average of vertices
        manual_center = jnp.mean(test_panel, axis=0)

        print("Single panel test:")
        print(f"  Computed center: {computed_center}")
        print(f"  Manual center: {manual_center}")
        print(f"  Difference: {jnp.linalg.norm(computed_center - manual_center)}")

        tolerance = 1e-12
        assert jnp.allclose(
            computed_center,
            manual_center,
            atol=tolerance,
        ), f"Single panel center mismatch: {jnp.linalg.norm(computed_center - manual_center)}"

        # Test vectorized computation
        vectorized_panel_center = vmap(panel_center)
        computed_centers = vectorized_panel_center(surface_panels)

        # Verify all centers are within panel bounds
        for i, (panel, center) in enumerate(zip(surface_panels, computed_centers)):
            # Check that center is within reasonable bounds of panel vertices
            min_coords = jnp.min(panel, axis=0)
            max_coords = jnp.max(panel, axis=0)

            within_bounds = jnp.all((center >= min_coords) & (center <= max_coords))
            assert within_bounds, f"Panel {i} center is outside panel bounds"

        print(f"Vectorized test ({len(surface_panels)} panels):")
        print("  All centers computed successfully")
        print("  All centers within panel bounds")

        print("✓ panel_center validation passed\n")

    def test_near_wake_panels_validation(self, test_plane: LSPT_Plane) -> None:
        """
        Test utility functions on near wake panels.

        Args:
            test_plane: LSPT_Plane object with precomputed data
        """
        print("Testing utility functions on near wake panels...")

        if test_plane.num_near_wake_panels == 0:
            print("  No near wake panels to test")
            return

        # Get near wake panels
        near_wake_panels = test_plane.panels[test_plane.near_wake_indices]
        precomputed_cps = test_plane.panel_cps[test_plane.near_wake_indices]
        precomputed_normals = test_plane.panel_normals[test_plane.near_wake_indices]

        # Test control points
        vectorized_panel_cp = vmap(panel_cp)
        computed_cps = vectorized_panel_cp(near_wake_panels)

        tolerance = 1e-6
        cp_match = jnp.allclose(computed_cps, precomputed_cps, atol=tolerance)

        # Test normals
        vectorized_panel_nj = vmap(panel_cp_normal)
        computed_normals = vectorized_panel_nj(near_wake_panels)

        normal_match = jnp.allclose(computed_normals, precomputed_normals, atol=tolerance)

        print(f"  Near wake panels tested: {len(near_wake_panels)}")
        print(f"  Control points match: {cp_match}")
        print(f"  Normals match: {normal_match}")

        assert cp_match, "Near wake panel control points don't match"
        assert normal_match, "Near wake panel normals don't match"

        print("✓ Near wake panels validation passed\n")


def run_validation_with_airplane(airplane: Airplane) -> None:
    """
    Run validation tests with a specific airplane configuration.

    Args:
        airplane: Airplane object to create LSPT_Plane from
    """
    from ICARUS.aero import LSPT_Plane

    print("Creating LSPT_Plane from airplane...")
    test_plane = LSPT_Plane(airplane)

    print("LSPT_Plane created:")
    print(f"  Name: {test_plane.name}")
    print(f"  Surfaces: {len(test_plane.surfaces)}")
    print(f"  Total panels: {test_plane.num_panels}")
    print(f"  Near wake panels: {test_plane.num_near_wake_panels}")
    print(f"  Flat wake panels: {test_plane.num_flat_wake_panels}")
    print()

    # Create test instance
    test_suite = TestUtilityFunctions()

    # Run all validation tests
    try:
        test_suite.test_panel_cp_validation(test_plane)
        test_suite.test_panel_control_nj_validation(test_plane)
        test_suite.test_panel_area_validation(test_plane)
        test_suite.test_panel_center_validation(test_plane)
        test_suite.test_near_wake_panels_validation(test_plane)

        print("🎉 All validation tests passed!")

    except AssertionError as e:
        print(f"❌ Validation failed: {e}")
        raise
    except Exception as e:
        print(f"💥 Unexpected error during validation: {e}")
        raise


def run_performance_comparison(airplane: Airplane, num_iterations: int = 1000) -> None:
    """
    Compare performance between utility functions and manual calculations.

    Args:
        airplane: Airplane object to create LSPT_Plane from
        num_iterations: Number of iterations for timing
    """
    import time

    from ICARUS.aero import LSPT_Plane

    print(f"Running performance comparison ({num_iterations} iterations)...")

    test_plane = LSPT_Plane(airplane)
    surface_panels = test_plane.panels[test_plane.panel_indices]

    # Test single panel performance
    test_panel = surface_panels[0]

    # Time JAX utility function
    start_time = time.time()
    for _ in range(num_iterations):
        _ = panel_cp(test_panel)
    jax_time = time.time() - start_time

    # Time manual calculation (converted to JAX for fair comparison)
    def manual_cp(panel):
        leading_edge_mid = (panel[0] + panel[1]) / 2
        trailing_edge_mid = (panel[2] + panel[3]) / 2
        return leading_edge_mid + 3 / 4 * (trailing_edge_mid - leading_edge_mid)

    start_time = time.time()
    for _ in range(num_iterations):
        _ = manual_cp(test_panel)
    manual_time = time.time() - start_time

    print(f"Single panel computation ({num_iterations} iterations):")
    print(f"  JAX utility function: {jax_time:.4f}s")
    print(f"  Manual calculation: {manual_time:.4f}s")
    print(f"  Speedup: {manual_time / jax_time:.2f}x")

    # Test vectorized performance
    vectorized_panel_cp = vmap(panel_cp)
    vectorized_manual_cp = vmap(manual_cp)

    start_time = time.time()
    for _ in range(100):  # Fewer iterations for vectorized test
        _ = vectorized_panel_cp(surface_panels)
    jax_vectorized_time = time.time() - start_time

    start_time = time.time()
    for _ in range(100):
        _ = vectorized_manual_cp(surface_panels)
    manual_vectorized_time = time.time() - start_time

    print(f"Vectorized computation ({len(surface_panels)} panels, 100 iterations):")
    print(f"  JAX utility function: {jax_vectorized_time:.4f}s")
    print(f"  Manual calculation: {manual_vectorized_time:.4f}s")
    print(f"  Speedup: {manual_vectorized_time / jax_vectorized_time:.2f}x")


if __name__ == "__main__":
    """
    Example usage of the validation script.

    To run this script, you need to provide an Airplane object.
    """
    print("Aerodynamic Utility Functions Validation Script")
    print("=" * 50)

    # This would be replaced with actual airplane loading code
    # For example:
    # from ICARUS.database import Database
    # DB = Database.get_instance()
    # airplane = DB.load_vehicle("your_airplane_name")
    # run_validation_with_airplane(airplane)

    print("To run validation, provide an Airplane object to run_validation_with_airplane()")
    print("Example:")
    print("  from ICARUS.database import Database")
    print("  DB = Database.get_instance()")
    print("  airplane = DB.load_vehicle('your_airplane')")
    print("  run_validation_with_airplane(airplane)")
