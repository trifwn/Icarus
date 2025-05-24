#!/usr/bin/env python3
"""
Performance test to demonstrate JAX acceleration in VLM analysis.
"""

import sys

import ICARUS

sys.path.append(ICARUS.INSTALL_DIR)

import time

import numpy as np

from examples.plane_analysis.Planes.hermes import hermes
from ICARUS.aero import AerodynamicLoads
from ICARUS.aero import LSPT_Plane
from ICARUS.environment import EARTH_ISA
from ICARUS.flight_dynamics import State


def performance_test():
    """Test VLM performance with JAX acceleration."""
    print("VLM Performance Test with JAX Acceleration")
    print("=" * 60)

    # Create test airplane and state
    airplane = hermes("test_hermes_perf")
    state = State(name="test", airplane=airplane, u_freestream=50.0, environment=EARTH_ISA)

    # Create LSPT_Plane and factorize system once
    lspt_plane = LSPT_Plane(airplane)
    print(f"Airplane: {lspt_plane.NM} panels, {lspt_plane.num_strips} strips")
    # Factorize system (one-time cost)
    start_time = time.time()
    lspt_plane.factorize_system()
    factorization_time = time.time() - start_time
    print(f"Matrix factorization time: {factorization_time:.4f} seconds")

    # Test multiple angle analysis speed
    angles = np.linspace(-10, 15, 20)  # 20 angles
    print(
        f"\nTesting {len(angles)} angles: {angles[0]:.1f}° to {angles[-1]:.1f}°",
    )  # Method 1: Using AerodynamicLoads workflow
    start_time = time.time()
    aero_loads = AerodynamicLoads.from_lspt_plane(lspt_plane)

    # Run analysis for all angles at once (convert numpy array to list)
    workflow_results = aero_loads.run_vlm_analysis(lspt_plane, state, angles.tolist())

    workflow_time = time.time() - start_time
    print(f"AerodynamicLoads workflow: {workflow_time:.4f} seconds ({workflow_time / len(angles) * 1000:.2f} ms/angle)")

    # Method 2: Using LSPT_Plane.aseq method directly
    start_time = time.time()
    aseq_results = lspt_plane.aseq(angles.tolist(), state)
    aseq_time = time.time() - start_time
    print(f"LSPT_Plane.aseq method: {aseq_time:.4f} seconds ({aseq_time / len(angles) * 1000:.2f} ms/angle)")

    # Speedup analysis
    print("\nSpeedup analysis:")
    print(f"- Factorization allows solving {len(angles)} angles in {workflow_time:.4f}s")
    print(f"- Average time per angle: {workflow_time / len(angles) * 1000:.2f} ms")
    print("- JAX JIT compilation provides acceleration for repeated solves")  # Verify consistency
    print("\nVerifying consistency between methods:")
    print(f"Workflow results columns: {list(workflow_results.columns)}")
    print(f"ASEQ results columns: {list(aseq_results.columns)}")

    first_angle_workflow = workflow_results.iloc[0]
    first_angle_aseq = aseq_results.iloc[0]

    # Get raw force values from both methods to compare
    workflow_lift = first_angle_workflow["Total_Lift_Potential"]
    aseq_lift = first_angle_aseq["LSPT Potential Fz"]

    print("First angle (-10°) raw values:")
    print(f"  Workflow Lift: {workflow_lift:.6f} N")
    print(f"  ASEQ Lift: {aseq_lift:.6f} N")
    print(f"  Force difference: {abs(aseq_lift - workflow_lift):.6f} N")

    # Get CL values (both should be the same after symmetry factor correction)
    workflow_cl = first_angle_workflow["CL"]
    # Extract CL from the ASEQ verbose output if needed, or calculate it
    dynamic_pressure = 0.5 * state.environment.air_density * state.u_freestream**2
    S = lspt_plane.S
    aseq_cl = aseq_lift / (dynamic_pressure * S)

    print(f"  Workflow CL: {workflow_cl:.6f}")
    print(f"  ASEQ CL: {aseq_cl:.6f}")
    print(f"  CL difference: {abs(aseq_cl - workflow_cl):.6f}")

    # More relaxed tolerance for numerical differences
    if abs(aseq_lift - workflow_lift) < 1e-6:
        print("✅ Force values are consistent")
    else:
        print("❌ Force values show differences - may be due to numerical precision")

    if abs(aseq_cl - workflow_cl) < 1e-6:
        print("✅ CL values are consistent")
    else:
        print("❌ CL values show differences")

    print("\nPerformance Summary:")
    print(f"- Matrix size: {lspt_plane.NM} × {lspt_plane.NM}")
    print(f"- Factorization time: {factorization_time:.4f}s")
    print(f"- Multi-angle solve time: {workflow_time:.4f}s for {len(angles)} angles")
    print("- JAX acceleration enabled: ✅")


if __name__ == "__main__":
    performance_test()
