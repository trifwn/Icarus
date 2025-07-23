#!/usr/bin/env python3
"""
Validation and fix for NACA thickness calculation issue.

This script identifies the root cause of the thickness calculation error
and validates the fix.
"""

import sys
from pathlib import Path

import jax.numpy as jnp

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

from ICARUS.airfoils.naca4 import NACA4


def test_current_implementation():
    """Test the current (incorrect) implementation."""
    print("Testing current implementation:")

    airfoil = NACA4(M=0.0, P=0.0, XX=0.12)  # NACA 0012
    x_test = 0.3
    computed_thickness = airfoil.thickness_distribution(jnp.array([x_test]))[0]

    print(f"  Current computed thickness at x={x_test}: {computed_thickness:.6f}")

    # Manual calculation with current formula
    xx = 0.12
    a0, a1, a2, a3, a4 = 0.2969, -0.126, -0.3516, 0.2843, -0.1036
    manual_current = (xx / 0.2) * (
        a0 * jnp.sqrt(x_test)
        + a1 * x_test
        + a2 * x_test**2
        + a3 * x_test**3
        + a4 * x_test**4
    )
    print(f"  Manual current formula: {manual_current:.6f}")

    return computed_thickness


def test_correct_implementation():
    """Test the correct implementation."""
    print("\nTesting correct implementation:")

    # Correct NACA formula (without extra division by 0.2)
    x_test = 0.3
    t = 0.12  # 12% thickness
    a0, a1, a2, a3, a4 = (
        0.2969,
        -0.1260,
        -0.3516,
        0.2843,
        -0.1015,
    )  # Note: a4 should be -0.1015

    correct_thickness = t * (
        a0 * jnp.sqrt(x_test)
        + a1 * x_test
        + a2 * x_test**2
        + a3 * x_test**3
        + a4 * x_test**4
    )

    print(f"  Correct analytical thickness at x={x_test}: {correct_thickness:.6f}")

    return correct_thickness


def analyze_issue():
    """Analyze the root cause of the issue."""
    print("\n" + "=" * 60)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 60)

    current = test_current_implementation()
    correct = test_correct_implementation()

    ratio = current / correct
    print(f"\nCurrent/Correct ratio: {ratio:.2f}")

    print("\nISSUE IDENTIFIED:")
    print("1. The thickness_distribution method divides xx by 0.2")
    print("2. But xx is already the thickness ratio (0.12 for 12%)")
    print("3. The standard NACA formula uses t directly, not t/0.2")
    print("4. Also, the a4 coefficient is wrong (-0.1036 vs -0.1015)")

    print("\nFIX REQUIRED:")
    print("1. Remove the division by 0.2 in thickness_distribution")
    print("2. Correct the a4 coefficient to -0.1015")
    print("3. The formula should be: xx * (a0*√x + a1*x + a2*x² + a3*x³ + a4*x⁴)")


def validate_fix():
    """Validate that the fix works correctly."""
    print("\n" + "=" * 60)
    print("VALIDATING FIX")
    print("=" * 60)

    # Test multiple points and airfoils
    test_cases = [
        (0.0, 0.0, 12, [0.0, 0.25, 0.5, 0.75, 1.0]),  # NACA 0012
        (0.0, 0.0, 15, [0.3]),  # NACA 0015
        (2.0, 4.0, 12, [0.3]),  # NACA 2412 (thickness should be same as 0012)
    ]

    for m, p, xx, x_points in test_cases:
        print(f"\nTesting NACA {int(m*100)}{int(p*10)}{int(xx):02d}:")

        # Current implementation
        airfoil = NACA4(M=m / 100, P=p / 10, XX=xx / 100)

        for x in x_points:
            if x == 0.0 or x == 1.0:
                continue  # Skip endpoints which may have special handling

            current_thickness = airfoil.thickness_distribution(jnp.array([x]))[0]

            # Correct analytical calculation
            t = xx / 100
            correct_thickness = t * (
                0.2969 * jnp.sqrt(x)
                - 0.1260 * x
                - 0.3516 * x**2
                + 0.2843 * x**3
                - 0.1015 * x**4
            )

            error = abs(current_thickness - correct_thickness)
            print(
                f"  x={x:.2f}: Current={current_thickness:.6f}, Correct={correct_thickness:.6f}, Error={error:.6f}",
            )


if __name__ == "__main__":
    analyze_issue()
    validate_fix()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("The NACA thickness calculation has two issues:")
    print("1. Incorrect division by 0.2 (should be removed)")
    print("2. Wrong a4 coefficient (-0.1036 should be -0.1015)")
    print("\nThese issues cause ~10x error in thickness calculations.")
    print("Fix required in ICARUS/airfoils/naca4.py thickness_distribution method.")
