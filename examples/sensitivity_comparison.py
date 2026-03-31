"""
Comparison: JAX Autodiff vs Finite Differences vs AVL

This script validates the JAX automatic differentiation results by comparing:
1. JAX autodiff stability derivatives (CL_alpha, CD_alpha, Cm_alpha)
2. Finite-difference derivatives from our own VLM
3. AVL solver results (if available)
"""

from __future__ import annotations

import os
import sys

import numpy as np

import jax
import jax.numpy as jnp

from ICARUS.airfoils import NACA4
from ICARUS.vehicle import Airplane, SymmetryAxes, WingSegment
from ICARUS.aero import LSPT_Plane
from ICARUS.aero.vlm.functional import make_vlm_force_fn, make_vlm_coefficients_fn


def create_test_airplane(N: int = 15, M: int = 8) -> tuple[Airplane, WingSegment]:
    """Create a simple rectangular wing for testing."""
    wing = WingSegment(
        name="benchmark",
        root_airfoil=NACA4.from_digits("4415"),
        origin=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0]),
        symmetries=SymmetryAxes.Y,
        span=2 * 5,  # 10m total
        sweep_offset=0.0,
        root_chord=1.0,
        tip_chord=1.0,
        N=N,
        M=M,
        mass=1.0,
    )
    airplane = Airplane(wing.name, main_wing=wing)
    return airplane, wing


def finite_difference_derivative(fn, x, h=1e-5):
    """Central finite difference approximation."""
    f_plus = fn(x + h)
    f_minus = fn(x - h)
    return (f_plus - f_minus) / (2 * h)


def compare_alpha_derivatives():
    """Compare CL_alpha, CD_alpha, Cm_alpha between autodiff and FD."""
    print("=" * 70)
    print("COMPARISON: JAX Autodiff vs Finite Differences")
    print("  Stability Derivatives w.r.t. Angle of Attack")
    print("=" * 70)

    airplane, wing = create_test_airplane()
    lspt_plane = LSPT_Plane(airplane)

    airspeed = 20.0
    density = 1.225
    alpha_test = 5.0

    # Create differentiable coefficient functions
    coeff_fn = make_vlm_coefficients_fn(lspt_plane, airspeed=airspeed, density=density)

    # Evaluate baseline
    CL, CD, Cm = coeff_fn(alpha_test)
    print(f"\nBaseline at alpha = {alpha_test} deg:")
    print(f"  CL = {float(CL):.6f}")
    print(f"  CD = {float(CD):.6f}")
    print(f"  Cm = {float(Cm):.6f}")

    # --- JAX Autodiff ---
    dcl_dalpha = float(jax.grad(lambda a: coeff_fn(a)[0])(alpha_test))
    dcd_dalpha = float(jax.grad(lambda a: coeff_fn(a)[1])(alpha_test))
    dcm_dalpha = float(jax.grad(lambda a: coeff_fn(a)[2])(alpha_test))

    # Convert to per-radian
    dcl_dalpha_rad = dcl_dalpha * 180 / np.pi
    dcd_dalpha_rad = dcd_dalpha * 180 / np.pi
    dcm_dalpha_rad = dcm_dalpha * 180 / np.pi

    print(f"\nJAX Autodiff (per radian):")
    print(f"  CL_alpha = {dcl_dalpha_rad:.6f}")
    print(f"  CD_alpha = {dcd_dalpha_rad:.6f}")
    print(f"  Cm_alpha = {dcm_dalpha_rad:.6f}")

    # --- Finite Differences ---
    for h_label, h in [("h=1e-3", 1e-3), ("h=1e-5", 1e-5), ("h=1e-7", 1e-7)]:
        cl_fd = finite_difference_derivative(lambda a: float(coeff_fn(a)[0]), alpha_test, h)
        cd_fd = finite_difference_derivative(lambda a: float(coeff_fn(a)[1]), alpha_test, h)
        cm_fd = finite_difference_derivative(lambda a: float(coeff_fn(a)[2]), alpha_test, h)

        cl_fd_rad = cl_fd * 180 / np.pi
        cd_fd_rad = cd_fd * 180 / np.pi
        cm_fd_rad = cm_fd * 180 / np.pi

        # Relative errors
        cl_err = abs(cl_fd_rad - dcl_dalpha_rad) / abs(dcl_dalpha_rad) * 100
        cd_err = abs(cd_fd_rad - dcd_dalpha_rad) / abs(dcd_dalpha_rad) * 100
        cm_err = abs(cm_fd_rad - dcm_dalpha_rad) / abs(dcm_dalpha_rad) * 100

        print(f"\nFinite Diff ({h_label}, per radian):")
        print(f"  CL_alpha = {cl_fd_rad:.6f}  (err = {cl_err:.2e}%)")
        print(f"  CD_alpha = {cd_fd_rad:.6f}  (err = {cd_err:.2e}%)")
        print(f"  Cm_alpha = {cm_fd_rad:.6f}  (err = {cm_err:.2e}%)")

    # --- Polar Sweep ---
    print("\n" + "=" * 70)
    print("POLAR SWEEP: LSPT VLM")
    print("=" * 70)
    angles = np.linspace(-2, 10, 13)
    print(f"\n{'AoA':>6s}  {'CL':>10s}  {'CD':>10s}  {'Cm':>10s}  {'CL_a (AD)':>10s}  {'CD_a (AD)':>10s}")
    print("-" * 70)

    for alpha in angles:
        cl, cd, cm = coeff_fn(float(alpha))
        dcl = float(jax.grad(lambda a: coeff_fn(a)[0])(float(alpha))) * 180 / np.pi
        dcd = float(jax.grad(lambda a: coeff_fn(a)[1])(float(alpha))) * 180 / np.pi
        print(f"{alpha:6.1f}  {float(cl):10.6f}  {float(cd):10.6f}  {float(cm):10.6f}  {dcl:10.4f}  {dcd:10.4f}")


def compare_with_avl():
    """Compare LSPT VLM results with AVL."""
    print("\n" + "=" * 70)
    print("COMPARISON: LSPT (JAX) vs AVL")
    print("=" * 70)

    try:
        from ICARUS.database import Database
        from ICARUS.environment.definition import EARTH_ISA
        from ICARUS.flight_dynamics import State
        from ICARUS.solvers.AVL import AVL

        # Initialize database
        db_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_data")
        os.makedirs(db_folder, exist_ok=True)
        Database._instance = None
        db = Database(db_folder)

        airplane, wing = create_test_airplane()
        state = State(
            name="Cruise",
            airplane=airplane,
            airspeed=20.0,
            environment=EARTH_ISA,
        )

        angles = np.linspace(-2, 10, 13)

        # Run AVL
        print("\nRunning AVL...")
        avl = AVL()
        try:
            avl.aseq(airplane, state, angles.tolist())
            avl_ran = True
            print("AVL completed successfully")
        except Exception as e:
            print(f"AVL failed: {e}")
            avl_ran = False

        if avl_ran and state.polar is not None:
            avl_df = state.polar

            # Run LSPT
            lspt_plane = LSPT_Plane(airplane)
            coeff_fn = make_vlm_coefficients_fn(
                lspt_plane,
                airspeed=state.airspeed,
                density=state.environment.air_density,
            )

            print(f"\n{'AoA':>6s}  {'LSPT CL':>10s}  {'AVL CL':>10s}  {'CL diff':>10s}  {'LSPT CD':>10s}  {'AVL CD':>10s}  {'CD diff':>10s}")
            print("-" * 80)

            for alpha in angles:
                cl_lspt, cd_lspt, _ = coeff_fn(float(alpha))

                # Try to get AVL values at this alpha
                avl_row = avl_df[avl_df["AoA"].round(1) == round(alpha, 1)]
                if len(avl_row) > 0:
                    cl_avl = avl_row.iloc[0].get("CL", float("nan"))
                    cd_avl = avl_row.iloc[0].get("CD", float("nan"))
                    cl_diff = float(cl_lspt) - cl_avl
                    cd_diff = float(cd_lspt) - cd_avl
                    print(
                        f"{alpha:6.1f}  {float(cl_lspt):10.6f}  {cl_avl:10.6f}  {cl_diff:10.6f}  "
                        f"{float(cd_lspt):10.6f}  {cd_avl:10.6f}  {cd_diff:10.6f}"
                    )
                else:
                    print(f"{alpha:6.1f}  {float(cl_lspt):10.6f}  {'N/A':>10s}  {'':>10s}  {float(cd_lspt):10.6f}")
        else:
            print("Skipping AVL comparison (no results available)")

    except ImportError as e:
        print(f"Could not import AVL dependencies: {e}")
    except Exception as e:
        print(f"AVL comparison failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    compare_alpha_derivatives()
    compare_with_avl()
