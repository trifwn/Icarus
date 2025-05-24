#!/usr/bin/env python3
"""
Test script to validate the VLM integration with AerodynamicLoads workflow.

This script tests the complete workflow:
1. Create LSPT_Plane from airplane
2. Create AerodynamicLoads from LSPT_Plane
3. Run VLM analysis with JAX acceleration
4. Validate results
"""

import sys

import ICARUS

sys.path.append(ICARUS.INSTALL_DIR)

import numpy as np

from ICARUS.aero import AerodynamicLoads
from ICARUS.aero import LSPT_Plane
from ICARUS.environment import EARTH_ISA
from ICARUS.flight_dynamics import State


def get_test_airplane_and_state():
    """Get a test airplane and flight state."""
    try:
        # Use the hermes airplane creation function
        from examples.plane_analysis.Planes.hermes import hermes

        # Create the hermes airplane
        airplane = hermes("test_hermes")

        # Create flight state
        state = State(
            name="test",
            airplane=airplane,
            u_freestream=50.0,  # 50 m/s
            environment=EARTH_ISA,
        )

        print(f"Created test airplane: {airplane.name}")
        print(f"  - Reference area: {airplane.S:.3f} m²")
        print(f"  - Number of surfaces: {len(airplane.surfaces)}")
        print(f"  - Total mass: {airplane.M:.3f} kg")
        return airplane, state

    except Exception as e:
        print(f"Error creating test airplane: {e}")
        return None, None


def test_lspt_plane_creation():
    """Test creating LSPT_Plane from airplane."""
    print("Testing LSPT_Plane creation...")

    airplane, state = get_test_airplane_and_state()
    if airplane is None or state is None:
        raise ValueError("Failed to load airplane and state from database")

    # Create LSPT_Plane
    lspt_plane = LSPT_Plane(airplane)

    print(f"  ✓ Created LSPT_Plane with {lspt_plane.NM} panels")
    print(f"  ✓ Number of strips: {lspt_plane.num_strips}")
    print(f"  ✓ Reference area: {lspt_plane.S} m²")

    return lspt_plane, state


def test_aerodynamic_loads_creation():
    """Test creating AerodynamicLoads from LSPT_Plane."""
    print("\nTesting AerodynamicLoads creation...")

    lspt_plane, state = test_lspt_plane_creation()

    # Create AerodynamicLoads from LSPT_Plane
    aero_loads = AerodynamicLoads.from_lspt_plane(lspt_plane)

    print(f"  ✓ Created AerodynamicLoads with {len(aero_loads)} strips")

    # Verify strips were copied correctly
    assert len(aero_loads.strips) == lspt_plane.num_strips
    print(f"  ✓ Strip count matches: {len(aero_loads.strips)}")

    return lspt_plane, aero_loads, state


def test_vlm_factorization():
    """Test VLM matrix factorization."""
    print("\nTesting VLM matrix factorization...")

    lspt_plane, aero_loads, state = test_aerodynamic_loads_creation()

    # Test factorization
    A_LU, A_piv, A_star = lspt_plane.factorize_system()

    print(f"  ✓ Factorized system with A shape: {A_LU.shape}")
    print(f"  ✓ A_star shape: {A_star.shape}")
    print(f"  ✓ Pivot vector length: {len(A_piv)}")

    # Verify factorization was stored
    assert hasattr(lspt_plane, "A_LU")
    assert hasattr(lspt_plane, "A_piv")
    assert hasattr(lspt_plane, "A_star")
    print("  ✓ Factorized matrices stored in plane object")

    return lspt_plane, aero_loads, state


def test_single_angle_analysis():
    """Test VLM analysis for a single angle."""
    print("\nTesting single angle VLM analysis...")

    lspt_plane, aero_loads, state = test_vlm_factorization()

    # Test single angle
    angle = 5.0  # 5 degrees AoA
    angles = [angle]

    # Run VLM analysis
    results_df = aero_loads.run_vlm_analysis(lspt_plane, state, angles)

    print(f"  ✓ Completed analysis for {angle}° AoA")
    print(f"  ✓ Results columns: {list(results_df.columns)}")

    # Print results
    row = results_df.iloc[0]
    print(f"  ✓ CL = {row['CL']:.4f}")
    print(f"  ✓ CD = {row['CD']:.6f}")
    print(f"  ✓ Total Lift = {row['Total_Lift_Combined']:.2f} N")
    print(f"  ✓ Total Drag = {row['Total_Drag_Combined']:.2f} N")

    # Basic sanity checks
    assert row["CL"] > 0, "CL should be positive for positive AoA"
    assert row["CD"] > 0, "CD should be positive"
    print("  ✓ Sanity checks passed")

    return lspt_plane, aero_loads, state, results_df


def test_multi_angle_analysis():
    """Test VLM analysis for multiple angles."""
    print("\nTesting multi-angle VLM analysis...")

    lspt_plane, aero_loads, state = test_vlm_factorization()

    # Test multiple angles
    angles = np.linspace(-5, 10, 8).tolist()  # Convert to list[float]

    print(f"  Testing angles: {angles}")

    # Run VLM analysis
    results_df = aero_loads.run_vlm_analysis(lspt_plane, state, angles)

    print(f"  ✓ Completed analysis for {len(angles)} angles")

    # Print summary
    print("\n  Results Summary:")
    print("  " + "=" * 60)
    print(f"  {'AoA (deg)':>8} {'CL':>8} {'CD':>8} {'L/D':>8}")
    print("  " + "-" * 60)

    for _, row in results_df.iterrows():
        ld_ratio = row["CL"] / row["CD"] if row["CD"] > 0 else 0
        print(f"  {row['AoA']:8.1f} {row['CL']:8.4f} {row['CD']:8.6f} {ld_ratio:8.2f}")

    # Check monotonic behavior
    assert results_df["CL"].is_monotonic_increasing, "CL should increase with AoA"
    print("  ✓ CL increases monotonically with AoA")

    return results_df


def test_aseq_method():
    """Test the aseq method directly on LSPT_Plane."""
    print("\nTesting LSPT_Plane.aseq method...")

    lspt_plane, aero_loads, state = test_vlm_factorization()

    # Test aseq method
    angles = [0.0, 2.0, 4.0, 6.0]  # Convert to list[float]
    results_df = lspt_plane.aseq(angles, state)

    print(f"  ✓ aseq completed for {len(angles)} angles")
    print(f"  ✓ Results columns: {list(results_df.columns)}")

    # Print results
    print("\n  ASEQ Results:")
    print("  " + "=" * 80)
    for _, row in results_df.iterrows():
        print(
            f"  AoA: {row['AoA']:6.1f}°, Fz: {row['LSPT Potential Fz']:8.2f} N, Fx: {row['LSPT Potential Fx']:8.4f} N",
        )

    return results_df


def test_component_separation():
    """Test separation of potential vs viscous loads."""
    print("\nTesting potential vs viscous load separation...")

    lspt_plane, aero_loads, state, _ = test_single_angle_analysis()

    angle = 5.0
    results_df = aero_loads.run_vlm_analysis(lspt_plane, state, [angle])
    row = results_df.iloc[0]

    # Check that components add up
    lift_potential = row["Total_Lift_Potential"]
    lift_viscous = row["Total_Lift_Viscous"]
    lift_combined = row["Total_Lift_Combined"]

    drag_potential = row["Total_Drag_Potential"]
    drag_viscous = row["Total_Drag_Viscous"]
    drag_combined = row["Total_Drag_Combined"]

    print(f"  Lift - Potential: {lift_potential:.2f} N, Viscous: {lift_viscous:.2f} N, Combined: {lift_combined:.2f} N")
    print(f"  Drag - Potential: {drag_potential:.4f} N, Viscous: {drag_viscous:.4f} N, Combined: {drag_combined:.4f} N")

    # Verify addition
    assert abs(lift_potential + lift_viscous - lift_combined) < 1e-6, "Lift components should add up"
    assert abs(drag_potential + drag_viscous - drag_combined) < 1e-6, "Drag components should add up"
    print("  ✓ Component addition verified")


def main():
    """Run all tests."""
    print("=" * 80)
    print("VLM Integration Test Suite")
    print("=" * 80)

    try:
        # Run tests in sequence
        test_lspt_plane_creation()
        test_aerodynamic_loads_creation()
        test_vlm_factorization()
        test_single_angle_analysis()
        test_multi_angle_analysis()
        test_aseq_method()
        test_component_separation()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("VLM integration with AerodynamicLoads is working correctly.")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()
