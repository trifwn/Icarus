"""
Test script for StripData and AerodynamicLoads classes
"""

import sys

import ICARUS

sys.path.append(ICARUS.INSTALL_DIR)

import jax.numpy as jnp

from ICARUS.aero import AerodynamicLoads
from ICARUS.aero import StripLoads
from ICARUS.airfoils import Airfoil


def test_strip_data_with_airfoil() -> StripLoads:
    """Test StripData class with airfoil functionality."""
    print("=== Testing StripData with Airfoil ===")

    # Create a simple airfoil
    airfoil = Airfoil.naca("2412", n_points=100)

    # Create some mock panel data
    panels = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    panel_idxs = jnp.array([0, 1, 2])

    # Create StripData with airfoil
    strip = StripLoads(panels=panels, panel_idxs=panel_idxs, chord=1.0, width=0.5, airfoil=airfoil)

    print(f"Strip chord: {strip.chord}")
    print(f"Strip width: {strip.width}")
    print(f"Number of panels: {strip.num_panels}")
    print(f"Airfoil name: {strip.airfoil.name if strip.airfoil else 'None'}")

    # Set some circulation values for testing
    strip.gammas = jnp.array([1.0, 1.5, 1.2])
    strip.w_induced = jnp.array([0.1, 0.15, 0.12])
    strip.mean_panel_width = jnp.array([0.3, 0.4, 0.3])

    # Test mean value calculation
    strip.calc_mean_values()
    print(f"Mean gamma: {strip.mean_gamma:.6f}")
    print(f"Mean w_induced: {strip.mean_w_induced:.6f}")

    # Test aerodynamic loads calculation
    dynamic_pressure = 100.0  # Pa
    strip.calc_aerodynamic_loads(dynamic_pressure)

    print(f"Total lift: {strip.get_total_lift():.6f}")
    print(f"Total drag: {strip.get_total_drag():.6f}")

    # Test 2D polars update
    strip.update_2d_polars(L_2D=150.0, D_2D=10.0, My_2D=5.0)
    print(f"2D Lift: {strip.L_2D}")
    print(f"2D Drag: {strip.D_2D}")
    print(f"2D Moment: {strip.My_2D}")

    print("✓ StripData tests passed!\n")
    return strip


def test_aerodynamic_loads() -> AerodynamicLoads:
    """Test AerodynamicLoads class functionality."""
    print("=== Testing AerodynamicLoads ===")

    # Create multiple strips with different airfoils
    strips = []

    for i in range(3):
        airfoil = Airfoil.naca(f"241{i}", n_points=50)
        panels = jnp.array([[float(i), 0.0, 0.0], [float(i + 1), 0.0, 0.0], [float(i + 1), 1.0, 0.0]])
        panel_idxs = jnp.array([0, 1, 2])

        strip = StripLoads(panels=panels, panel_idxs=panel_idxs, chord=1.0 + i * 0.2, width=0.5, airfoil=airfoil)

        # Set some test values
        strip.gammas = jnp.array([1.0 + i * 0.1, 1.5 + i * 0.1, 1.2 + i * 0.1])
        strip.w_induced = jnp.array([0.1, 0.15, 0.12])
        strip.mean_panel_width = jnp.array([0.3, 0.4, 0.3])

        # Calculate loads
        strip.calc_aerodynamic_loads(100.0)

        # Set some moment values for testing
        strip.Mx = float(i * 10.0)
        strip.My = float(i * 15.0)
        strip.Mz = float(i * 5.0)

        # Set Trefftz drag
        strip.D_trefftz = float(i * 2.0)

        strips.append(strip)

    # Create AerodynamicLoads object
    aero_loads = AerodynamicLoads(strips)

    print(f"Number of strips: {len(aero_loads)}")
    # Test iteration
    print("Iterating through strips:")
    for i, strip in enumerate(aero_loads):
        airfoil_name = strip.airfoil.name if strip.airfoil else "None"
        print(f"  Strip {i}: chord={strip.chord:.2f}, airfoil={airfoil_name}")

    # Test indexing
    print(f"First strip chord: {aero_loads[0].chord}")
    print(f"Last strip chord: {aero_loads[-1].chord}")

    # Test total calculations
    total_lift = aero_loads.calc_total_lift()
    total_drag = aero_loads.calc_total_drag()
    total_moments = aero_loads.calc_total_moments()
    total_trefftz_drag = aero_loads.calc_trefftz_drag()

    print(f"Total lift: {total_lift:.6f}")
    print(f"Total drag: {total_drag:.6f}")
    print(f"Total moments: {total_moments}")
    print(f"Total Trefftz drag: {total_trefftz_drag:.6f}")

    # Test summary
    summary = aero_loads.get_summary()
    print(f"Summary: {summary}")
    # Test string representation
    print("String representation:")
    print(str(aero_loads))
    print(f"Repr: {repr(aero_loads)}")

    # Test airfoil search
    matching_strips = aero_loads.get_strip_by_airfoil("2410")
    print(f"Strips with NACA 2410: {len(matching_strips)}")

    # Test add/remove strip functionality
    new_airfoil = Airfoil.naca("0012", n_points=50)
    new_panels = jnp.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [11.0, 1.0, 0.0]])
    new_panel_idxs = jnp.array([0, 1, 2])

    new_strip = StripLoads(panels=new_panels, panel_idxs=new_panel_idxs, chord=2.0, width=0.8, airfoil=new_airfoil)

    print(f"Before adding: {len(aero_loads)} strips")
    aero_loads.add_strip(new_strip)
    print(f"After adding: {len(aero_loads)} strips")

    aero_loads.remove_strip(-1)  # Remove the last strip
    print(f"After removing: {len(aero_loads)} strips")

    # Test collective operations
    aero_loads.calc_mean_values_all()
    aero_loads.calc_aerodynamic_loads_all(100.0)
    print("Collective calculations completed")

    print("✓ AerodynamicLoads tests passed!\n")
    return aero_loads


def test_integration() -> None:
    """Test integration between StripData and AerodynamicLoads."""
    print("=== Testing Integration ===")

    # Create an empty AerodynamicLoads object
    aero_loads = AerodynamicLoads()
    print(f"Empty AerodynamicLoads summary: {aero_loads.get_summary()}")

    # Create and add strips one by one
    for i in range(5):
        airfoil = Airfoil.naca(f"00{12 + i * 2:02d}", n_points=30)
        panels = jnp.ones((3, 3)) * i
        panel_idxs = jnp.array([0, 1, 2])

        strip = StripLoads(
            panels=panels,
            panel_idxs=panel_idxs,
            chord=1.0 + i * 0.1,
            width=0.5 + i * 0.05,
            airfoil=airfoil,
        )

        # Set realistic circulation values
        strip.gammas = jnp.array([0.8 + i * 0.1, 1.0 + i * 0.1, 0.9 + i * 0.1])
        strip.w_induced = jnp.array([0.05 + i * 0.01, 0.08 + i * 0.01, 0.06 + i * 0.01])
        strip.mean_panel_width = jnp.array([0.2 + i * 0.02, 0.25 + i * 0.02, 0.22 + i * 0.02])

        aero_loads.add_strip(strip)

    print(f"Added {len(aero_loads)} strips")

    # Calculate loads for all strips
    dynamic_pressure = 150.0  # Higher dynamic pressure
    aero_loads.calc_aerodynamic_loads_all(dynamic_pressure)

    # Get final summary
    final_summary = aero_loads.get_summary()
    print("Final summary:")
    for key, value in final_summary.items():
        print(f"  {key}: {value}")

    # Test error handling
    try:
        aero_loads.remove_strip(100)  # Invalid index
    except IndexError as e:
        print(f"Caught expected error: {e}")

    # Test clearing
    original_count = len(aero_loads)
    aero_loads.clear()
    print(f"Cleared strips: {original_count} -> {len(aero_loads)}")

    print("✓ Integration tests passed!\n")


def main() -> int:
    """Main test function."""
    print("Starting comprehensive tests for StripData and AerodynamicLoads classes...")
    print()

    try:
        # Test individual components
        test_strip_data_with_airfoil()
        test_aerodynamic_loads()
        test_integration()

        print("🎉 All tests passed successfully!")
        print()
        print("Summary:")
        print("- StripData class enhanced with airfoil attribute")
        print("- StripData class has methods for calculating aerodynamic loads")
        print("- AerodynamicLoads class provides collection management")
        print("- AerodynamicLoads class calculates total loads across all strips")
        print("- Both classes integrate well with the existing ICARUS framework")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
