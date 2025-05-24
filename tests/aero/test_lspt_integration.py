"""
Test script to verify LSPT integration with enhanced StripData
"""

import sys

import ICARUS

sys.path.append(ICARUS.INSTALL_DIR)

import jax.numpy as jnp

from ICARUS.aero import AerodynamicLoads
from ICARUS.aero import StripLoads
from ICARUS.airfoils import Airfoil


def test_lspt_integration():
    """Test that existing LSPT code can work with enhanced StripData."""
    print("=== Testing LSPT Integration ===")

    # Create StripData objects as they would be created in LSPT_Plane
    strips = []

    # Simulate what happens in LSPT_Plane.__init__
    for i in range(3):
        # Create mock panel data
        panels = jnp.array(
            [[float(i), 0.0, 0.0], [float(i + 1), 0.0, 0.0], [float(i + 1), 1.0, 0.0], [float(i), 1.0, 0.0]],
        )
        panel_idxs = jnp.array([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3])

        # Create airfoil (optional in existing code)
        airfoil = Airfoil.naca(f"241{i}", n_points=50)

        # Test both old-style creation (without airfoil) and new-style (with airfoil)
        if i == 0:
            # Old style - should still work
            strip = StripLoads(panels=panels, panel_idxs=panel_idxs, chord=1.0 + i * 0.2, width=0.5)
            print(f"Strip {i} created without airfoil: {strip.airfoil is None}")
        else:
            # New style - with airfoil
            strip = StripLoads(panels=panels, panel_idxs=panel_idxs, chord=1.0 + i * 0.2, width=0.5, airfoil=airfoil)
            print(f"Strip {i} created with airfoil: {strip.airfoil.name if strip.airfoil else 'None'}")

        # Verify all expected attributes exist
        assert hasattr(strip, "chord")
        assert hasattr(strip, "width")
        assert hasattr(strip, "panels")
        assert hasattr(strip, "panel_idxs")
        assert hasattr(strip, "num_panels")
        assert hasattr(strip, "gammas")
        assert hasattr(strip, "w_induced")
        assert hasattr(strip, "Ls")
        assert hasattr(strip, "Ds")
        assert hasattr(strip, "L_2D")
        assert hasattr(strip, "D_2D")
        assert hasattr(strip, "My_2D")
        assert hasattr(strip, "airfoil")  # New attribute

        # Test that existing methods still work
        strip.calc_mean_values()

        strips.append(strip)

    print(f"Created {len(strips)} strips successfully")

    # Test that AerodynamicLoads works with mixed strip types
    aero_loads = AerodynamicLoads(strips)

    # Set some circulation values to test calculations
    for i, strip in enumerate(aero_loads):
        strip.gammas = jnp.array([1.0 + i * 0.1] * strip.num_panels)
        strip.w_induced = jnp.array([0.1 + i * 0.01] * strip.num_panels)
        strip.mean_panel_width = jnp.array([0.25] * strip.num_panels)

    # Calculate loads
    aero_loads.calc_aerodynamic_loads_all(100.0)

    # Verify results
    total_lift = aero_loads.calc_total_lift()
    total_drag = aero_loads.calc_total_drag()

    print(f"Total lift across all strips: {total_lift:.6f}")
    print(f"Total drag across all strips: {total_drag:.6f}")

    # Test airfoil search functionality
    strips_with_airfoils = [s for s in aero_loads if s.airfoil is not None]
    strips_without_airfoils = [s for s in aero_loads if s.airfoil is None]

    print(f"Strips with airfoils: {len(strips_with_airfoils)}")
    print(f"Strips without airfoils: {len(strips_without_airfoils)}")

    # Search for specific airfoil
    naca2411_strips = aero_loads.get_strip_by_airfoil("naca2411")
    print(f"Strips with NACA 2411: {len(naca2411_strips)}")

    print("✓ LSPT integration tests passed!")
    return True


def test_backward_compatibility():
    """Test that existing code patterns still work."""
    print("\n=== Testing Backward Compatibility ===")

    # Test the old constructor signature still works
    panels = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    panel_idxs = jnp.array([0, 1, 2])

    # This is how StripData was created before
    old_style_strip = StripLoads(panels=panels, panel_idxs=panel_idxs, chord=1.5, width=0.8)

    # Verify it has the airfoil attribute but it's None
    assert hasattr(old_style_strip, "airfoil")
    assert old_style_strip.airfoil is None

    # Test all existing functionality still works
    old_style_strip.calc_mean_values()
    old_style_strip.calc_aerodynamic_loads(50.0)

    lift = old_style_strip.get_total_lift()
    drag = old_style_strip.get_total_drag()

    print(f"Old-style strip lift: {lift:.6f}")
    print(f"Old-style strip drag: {drag:.6f}")

    # Test that it can be added to AerodynamicLoads
    aero_loads = AerodynamicLoads()
    aero_loads.add_strip(old_style_strip)

    assert len(aero_loads) == 1
    assert aero_loads[0].airfoil is None

    print("✓ Backward compatibility tests passed!")
    return True


def main():
    """Main test function."""
    print("Testing LSPT integration and backward compatibility...")
    print()

    try:
        test_lspt_integration()
        test_backward_compatibility()

        print("\n🎉 All integration tests passed!")
        print()
        print("Integration Summary:")
        print("- Existing LSPT code can create StripData without airfoils")
        print("- New code can create StripData with airfoils")
        print("- All existing StripData functionality preserved")
        print("- AerodynamicLoads works with mixed strip types")
        print("- Backward compatibility maintained")

        return 0

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
