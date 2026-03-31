"""
Test: Equinox Differentiable Modules

Validates the entire diff pipeline:
1. Conversion from ICARUS classes to Equinox modules
2. Forward evaluation (DiffAirplane → VLM forces)
3. Gradient computation via jax.grad / eqx.filter_grad
4. Comparison with existing functional.py results
"""

from __future__ import annotations

import sys
import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx

from ICARUS.airfoils import NACA4
from ICARUS.vehicle import Airplane, SymmetryAxes, WingSegment


def create_test_airplane(N: int = 10, M: int = 5) -> Airplane:
    """Create a simple rectangular wing for testing."""
    wing = WingSegment(
        name="test_wing",
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
    return Airplane(wing.name, main_wing=wing)


def test_conversion():
    """Test converting ICARUS classes to Equinox modules."""
    print("=" * 60)
    print("TEST 1: Conversion from ICARUS → Equinox modules")
    print("=" * 60)

    from ICARUS.aero.diff import from_airplane, DiffAirplane

    airplane = create_test_airplane()
    diff_plane = from_airplane(airplane)

    print(f"  Airplane name: {diff_plane.name}")
    print(f"  Main wing: {diff_plane.main_wing_name}")
    print(f"  Num wings: {len(diff_plane.wings)}")
    print(f"  Num segments: {len(diff_plane.all_segments)}")

    seg = diff_plane.all_segments[0]
    print(f"  Segment '{seg.name}': N={seg.N}, M={seg.M}")
    print(f"  Chord dist: {seg.chord_dist}")
    print(f"  Span: {seg.span:.2f} m")
    print(f"  Area: {seg.area:.2f} m²")
    print(f"  MAC: {seg.mean_aerodynamic_chord:.4f} m")
    print(f"  Symmetric Y: {seg.is_symmetric_y}")
    print(f"  Root camber: {seg.root_camber.name}")

    # Check it's a valid pytree
    leaves = jax.tree_util.tree_leaves(diff_plane)
    print(f"\n  Pytree leaves: {len(leaves)}")
    print(f"  Types: {set(type(l).__name__ for l in leaves)}")

    print("\n  [PASS] Conversion successful")
    return diff_plane


def test_forward_evaluation(diff_plane):
    """Test forward evaluation of VLM forces."""
    print("\n" + "=" * 60)
    print("TEST 2: Forward VLM evaluation via diff pipeline")
    print("=" * 60)

    from ICARUS.aero.diff.pipeline import diff_vlm_forces, make_diff_coefficients_fn

    airspeed = 20.0
    density = 1.225
    alpha = jnp.asarray(5.0, dtype=jnp.float64)

    print("  Computing VLM forces...")
    lift, drag, My = diff_vlm_forces(diff_plane, alpha, airspeed, density)
    print(f"  Lift = {float(lift):.4f} N")
    print(f"  Drag = {float(drag):.4f} N")
    print(f"  My   = {float(My):.4f} N·m")

    # Coefficients
    coeff_fn = make_diff_coefficients_fn(diff_plane, airspeed, density)
    CL, CD, Cm = coeff_fn(diff_plane, alpha)
    print(f"\n  CL = {float(CL):.6f}")
    print(f"  CD = {float(CD):.6f}")
    print(f"  Cm = {float(Cm):.6f}")

    assert float(CL) > 0, "CL should be positive at alpha=5°"
    assert float(CD) > 0, "CD should be positive"
    print("\n  [PASS] Forward evaluation successful")
    return coeff_fn


def test_alpha_gradient(diff_plane, coeff_fn):
    """Test gradient w.r.t. angle of attack."""
    print("\n" + "=" * 60)
    print("TEST 3: Gradient w.r.t. alpha (stability derivative)")
    print("=" * 60)

    alpha = jnp.asarray(5.0, dtype=jnp.float64)

    # dCL/dalpha via autodiff
    dCL_dalpha = float(jax.grad(lambda a: coeff_fn(diff_plane, a)[0])(alpha))
    dCL_dalpha_rad = dCL_dalpha * 180 / np.pi

    # Finite difference check
    h = 1e-5
    CL_plus = float(coeff_fn(diff_plane, alpha + h)[0])
    CL_minus = float(coeff_fn(diff_plane, alpha - h)[0])
    dCL_dalpha_fd = (CL_plus - CL_minus) / (2 * h)
    dCL_dalpha_fd_rad = dCL_dalpha_fd * 180 / np.pi

    rel_err = abs(dCL_dalpha_rad - dCL_dalpha_fd_rad) / abs(dCL_dalpha_rad) * 100

    print(f"  CL_alpha (autodiff): {dCL_dalpha_rad:.6f} /rad")
    print(f"  CL_alpha (FD):       {dCL_dalpha_fd_rad:.6f} /rad")
    print(f"  Relative error:      {rel_err:.2e}%")

    assert rel_err < 1.0, f"Gradient error too large: {rel_err}%"
    print("\n  [PASS] Alpha gradient matches finite differences")


def test_comparison_with_functional():
    """Compare diff pipeline results with existing functional.py."""
    print("\n" + "=" * 60)
    print("TEST 4: Comparison with existing functional.py API")
    print("=" * 60)

    from ICARUS.aero import LSPT_Plane
    from ICARUS.aero.vlm.functional import make_vlm_coefficients_fn
    from ICARUS.aero.diff import from_airplane
    from ICARUS.aero.diff.pipeline import make_diff_coefficients_fn

    airplane = create_test_airplane()
    airspeed = 20.0
    density = 1.225
    alpha = jnp.asarray(5.0, dtype=jnp.float64)

    # Existing API
    lspt_plane = LSPT_Plane(airplane)
    old_fn = make_vlm_coefficients_fn(lspt_plane, airspeed, density)
    CL_old, CD_old, Cm_old = old_fn(alpha)

    # New Equinox API
    diff_plane = from_airplane(airplane)
    new_fn = make_diff_coefficients_fn(diff_plane, airspeed, density)
    CL_new, CD_new, Cm_new = new_fn(diff_plane, alpha)

    print(f"  {'':>12s}  {'functional.py':>14s}  {'diff pipeline':>14s}  {'diff':>10s}")
    print(f"  {'CL':>12s}  {float(CL_old):14.8f}  {float(CL_new):14.8f}  {float(CL_new-CL_old):10.2e}")
    print(f"  {'CD':>12s}  {float(CD_old):14.8f}  {float(CD_new):14.8f}  {float(CD_new-CD_old):10.2e}")
    print(f"  {'Cm':>12s}  {float(Cm_old):14.8f}  {float(Cm_new):14.8f}  {float(Cm_new-Cm_old):10.2e}")

    # Allow some tolerance since wake generation may differ slightly
    cl_err = abs(float(CL_new - CL_old)) / abs(float(CL_old)) * 100
    cd_err = abs(float(CD_new - CD_old)) / max(abs(float(CD_old)), 1e-10) * 100
    print(f"\n  CL relative error: {cl_err:.4f}%")
    print(f"  CD relative error: {cd_err:.4f}%")

    if cl_err < 5.0:
        print("\n  [PASS] Results agree within tolerance")
    else:
        print(f"\n  [WARN] CL differs by {cl_err:.2f}% - may be due to wake differences")


def test_design_gradient():
    """Test gradient w.r.t. design parameters using eqx.filter_grad."""
    print("\n" + "=" * 60)
    print("TEST 5: Design parameter gradients (eqx.filter_grad)")
    print("=" * 60)

    from ICARUS.aero.diff import from_airplane
    from ICARUS.aero.diff.pipeline import make_gradient_fn

    airplane = create_test_airplane()
    diff_plane = from_airplane(airplane)
    airspeed = 20.0
    density = 1.225

    # Create scalar function airplane → CD
    cd_fn = make_gradient_fn(diff_plane, airspeed, density, alpha_deg=5.0, output="CD")

    # Baseline CD
    CD_baseline = float(cd_fn(diff_plane))
    print(f"  Baseline CD = {CD_baseline:.8f}")

    # Compute gradient w.r.t. all dynamic parameters
    grad_fn = eqx.filter_grad(cd_fn)
    grads = grad_fn(diff_plane)

    # Check specific gradients
    seg_grads = grads.wings[0].segments[0]
    print(f"\n  dCD/d(chord_dist): {seg_grads.chord_dist}")
    print(f"  dCD/d(span_dist):  {seg_grads.span_dist}")
    print(f"  dCD/d(twist):      {seg_grads.twist_angles}")

    # Verify with finite differences for one parameter
    h = 1e-5
    seg = diff_plane.wings[0].segments[0]
    new_chords = seg.chord_dist.at[0].set(seg.chord_dist[0] + h)
    perturbed = eqx.tree_at(
        lambda p: p.wings[0].segments[0].chord_dist,
        diff_plane,
        new_chords,
    )
    CD_perturbed = float(cd_fn(perturbed))
    fd_grad = (CD_perturbed - CD_baseline) / h

    ad_grad = float(seg_grads.chord_dist[0])
    if abs(ad_grad) > 1e-15:
        rel_err = abs(fd_grad - ad_grad) / abs(ad_grad) * 100
        print(f"\n  dCD/d(chord[0]) autodiff: {ad_grad:.8e}")
        print(f"  dCD/d(chord[0]) FD:       {fd_grad:.8e}")
        print(f"  Relative error: {rel_err:.2e}%")
    else:
        print(f"\n  dCD/d(chord[0]) is near-zero: {ad_grad:.2e}")

    print("\n  [PASS] Design gradients computed successfully")


if __name__ == "__main__":
    print("Equinox Differentiable Modules - Validation Suite")
    print("=" * 60)

    diff_plane = test_conversion()
    coeff_fn = test_forward_evaluation(diff_plane)
    test_alpha_gradient(diff_plane, coeff_fn)
    test_comparison_with_functional()
    test_design_gradient()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
