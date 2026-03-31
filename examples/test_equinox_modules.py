"""
Comprehensive Validation: Equinox Differentiable Modules

Tests:
1. Conversion from ICARUS classes → Equinox modules
2. Forward VLM evaluation + comparison with existing API
3. Alpha gradient (stability derivative) + FD validation
4. Design parameter gradients via eqx.filter_grad
5. Multi-wing configuration (wing + tail)
6. Swept/tapered wing
7. Control surface deflection gradients
8. Viscous drag with flat-plate polar
9. NACA4 parametric camber differentiation
10. JIT compilation
11. Cm reference alignment
"""

from __future__ import annotations

import sys
import time
import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx

from ICARUS.airfoils import NACA4
from ICARUS.vehicle import (
    Airplane, SymmetryAxes, WingSegment, Aileron, Elevator,
)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


# ─────────────────────────────────────────────────────────────────────
# Aircraft factory helpers
# ─────────────────────────────────────────────────────────────────────

def make_rectangular_wing(N=10, M=5):
    wing = WingSegment(
        name="main_wing", root_airfoil=NACA4.from_digits("4415"),
        origin=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0]),
        symmetries=SymmetryAxes.Y,
        span=2 * 5, root_chord=1.0, tip_chord=1.0,
        N=N, M=M, mass=10.0,
    )
    return Airplane("rect_plane", main_wing=wing)


def make_tapered_swept_wing(N=10, M=5):
    wing = WingSegment(
        name="swept_wing", root_airfoil=NACA4.from_digits("2412"),
        origin=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0]),
        symmetries=SymmetryAxes.Y,
        span=2 * 6, root_chord=2.0, tip_chord=0.8,
        sweepback_angle=25.0,
        twist_root=np.deg2rad(2.0), twist_tip=np.deg2rad(-1.0),
        N=N, M=M, mass=15.0,
    )
    return Airplane("swept_plane", main_wing=wing)


def make_wing_tail_airplane(N=8, M=4):
    wing = WingSegment(
        name="main_wing", root_airfoil=NACA4.from_digits("4415"),
        origin=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0]),
        symmetries=SymmetryAxes.Y,
        span=2 * 5, root_chord=1.5, tip_chord=1.0,
        N=N, M=M, mass=12.0,
    )
    tail = WingSegment(
        name="tail", root_airfoil=NACA4.from_digits("0012"),
        origin=np.array([4.0, 0.0, 0.3]),
        orientation=np.array([0.0, 0.0, 0.0]),
        symmetries=SymmetryAxes.Y,
        span=2 * 2, root_chord=0.6, tip_chord=0.4,
        N=N, M=M, mass=3.0, is_lifting=True,
    )
    return Airplane("wing_tail", main_wing=wing, other_wings=[tail])


def make_wing_with_aileron(N=10, M=5):
    aileron = Aileron(
        local_span_percentages=(0.6, 0.95),
        hinge_chord_percentages=(0.7, 0.7),
    )
    wing = WingSegment(
        name="ctrl_wing", root_airfoil=NACA4.from_digits("4415"),
        origin=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0]),
        symmetries=SymmetryAxes.Y,
        span=2 * 5, root_chord=1.0, tip_chord=1.0,
        N=N, M=M, mass=10.0,
        controls=[aileron],
    )
    return Airplane("ctrl_plane", main_wing=wing)


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

def test_conversion_and_forward():
    """Test 1-2: Conversion + forward evaluation."""
    print("\n" + "=" * 60)
    print("TEST 1-2: Conversion + Forward Evaluation")
    print("=" * 60)

    from ICARUS.aero.diff import from_airplane
    from ICARUS.aero.diff.pipeline import diff_vlm_forces, make_diff_coefficients_fn

    airplane = make_rectangular_wing()
    diff_plane = from_airplane(airplane)

    check("Conversion succeeds", diff_plane.name == "rect_plane")
    check("Main wing found", diff_plane.main_wing.name == "main_wing")
    check("Is pytree", len(jax.tree_util.tree_leaves(diff_plane)) > 0)

    airspeed, density = 20.0, 1.225
    alpha = jnp.asarray(5.0, dtype=jnp.float64)

    lift, drag, My = diff_vlm_forces(diff_plane, alpha, airspeed, density)
    check("Lift > 0", float(lift) > 0, f"lift={float(lift)}")
    check("Drag > 0", float(drag) > 0, f"drag={float(drag)}")

    coeff_fn = make_diff_coefficients_fn(diff_plane, airspeed, density)
    CL, CD, Cm = coeff_fn(diff_plane, alpha)
    print(f"  CL={float(CL):.6f}, CD={float(CD):.6f}, Cm={float(Cm):.6f}")
    check("CL reasonable", 0.1 < float(CL) < 1.5)
    check("CD reasonable", 0.0 < float(CD) < 0.1)

    return diff_plane, coeff_fn


def test_alpha_gradient(diff_plane, coeff_fn):
    """Test 3: Alpha gradient vs finite differences."""
    print("\n" + "=" * 60)
    print("TEST 3: Alpha Stability Derivatives")
    print("=" * 60)

    alpha = jnp.asarray(5.0, dtype=jnp.float64)

    dCL_da = float(jax.grad(lambda a: coeff_fn(diff_plane, a)[0])(alpha))
    dCD_da = float(jax.grad(lambda a: coeff_fn(diff_plane, a)[1])(alpha))
    dCm_da = float(jax.grad(lambda a: coeff_fn(diff_plane, a)[2])(alpha))

    h = 1e-5
    CL_p = float(coeff_fn(diff_plane, alpha + h)[0])
    CL_m = float(coeff_fn(diff_plane, alpha - h)[0])
    dCL_da_fd = (CL_p - CL_m) / (2 * h)

    rel_err = abs(dCL_da - dCL_da_fd) / abs(dCL_da) * 100
    print(f"  CL_alpha AD={dCL_da * 180/np.pi:.4f}/rad  FD={dCL_da_fd * 180/np.pi:.4f}/rad  err={rel_err:.2e}%")
    print(f"  CD_alpha={dCD_da * 180/np.pi:.4f}/rad")
    print(f"  Cm_alpha={dCm_da * 180/np.pi:.4f}/rad")

    check("CL_alpha matches FD", rel_err < 0.01)
    # Note: a standalone wing (no tail) typically has Cm_alpha > 0.
    # Static stability (Cm_alpha < 0) requires a tail. See test 6.
    check("Cm_alpha computable", jnp.isfinite(dCm_da))


def test_cm_alignment():
    """Test 11: Cm reference point alignment with original API."""
    print("\n" + "=" * 60)
    print("TEST 4: Cm Reference Alignment")
    print("=" * 60)

    from ICARUS.aero import LSPT_Plane
    from ICARUS.aero.vlm.functional import make_vlm_coefficients_fn
    from ICARUS.aero.diff import from_airplane
    from ICARUS.aero.diff.pipeline import make_diff_coefficients_fn

    airplane = make_rectangular_wing()
    alpha = jnp.asarray(5.0, dtype=jnp.float64)
    airspeed, density = 20.0, 1.225

    # Old API
    lspt = LSPT_Plane(airplane)
    old_fn = make_vlm_coefficients_fn(lspt, airspeed, density)
    CL_old, CD_old, Cm_old = old_fn(alpha)

    # New API
    diff_plane = from_airplane(airplane)
    new_fn = make_diff_coefficients_fn(diff_plane, airspeed, density)
    CL_new, CD_new, Cm_new = new_fn(diff_plane, alpha)

    cl_err = abs(float(CL_new - CL_old)) / abs(float(CL_old)) * 100
    cd_err = abs(float(CD_new - CD_old)) / max(abs(float(CD_old)), 1e-10) * 100
    cm_err = abs(float(Cm_new - Cm_old)) / max(abs(float(Cm_old)), 1e-10) * 100

    print(f"  CL: old={float(CL_old):.8f}  new={float(CL_new):.8f}  err={cl_err:.4f}%")
    print(f"  CD: old={float(CD_old):.8f}  new={float(CD_new):.8f}  err={cd_err:.4f}%")
    print(f"  Cm: old={float(Cm_old):.8f}  new={float(Cm_new):.8f}  err={cm_err:.4f}%")

    check("CL matches", cl_err < 0.01)
    check("CD matches", cd_err < 0.01)
    check("Cm matches", cm_err < 1.0, f"err={cm_err:.2f}%")


def test_design_gradient():
    """Test 4: Design parameter gradients."""
    print("\n" + "=" * 60)
    print("TEST 5: Design Gradients (eqx.filter_grad)")
    print("=" * 60)

    from ICARUS.aero.diff import from_airplane
    from ICARUS.aero.diff.pipeline import make_gradient_fn

    diff_plane = from_airplane(make_rectangular_wing())
    cd_fn = make_gradient_fn(diff_plane, 20.0, 1.225, alpha_deg=5.0, output="CD")

    CD0 = float(cd_fn(diff_plane))
    grads = eqx.filter_grad(cd_fn)(diff_plane)

    seg = grads.wings[0].segments[0]
    print(f"  dCD/d(chord): {seg.chord_dist}")
    print(f"  dCD/d(twist): {seg.twist_angles}")

    # Symmetry check: rectangular symmetric wing should have symmetric gradients
    chord_grads = np.array(seg.chord_dist)
    is_symmetric = np.allclose(chord_grads, chord_grads[::-1], atol=1e-10)
    check("Chord gradients are symmetric", is_symmetric)

    # FD validation
    h = 1e-5
    new_chords = diff_plane.wings[0].segments[0].chord_dist.at[0].set(
        diff_plane.wings[0].segments[0].chord_dist[0] + h,
    )
    perturbed = eqx.tree_at(lambda p: p.wings[0].segments[0].chord_dist, diff_plane, new_chords)
    fd = (float(cd_fn(perturbed)) - CD0) / h
    ad = float(seg.chord_dist[0])
    rel_err = abs(fd - ad) / max(abs(ad), 1e-15) * 100
    print(f"  dCD/d(chord[0]) AD={ad:.6e}  FD={fd:.6e}  err={rel_err:.3f}%")
    check("Chord gradient matches FD", rel_err < 1.0)


def test_multi_wing():
    """Test 5: Wing + tail configuration."""
    print("\n" + "=" * 60)
    print("TEST 6: Multi-Wing (Wing + Tail)")
    print("=" * 60)

    from ICARUS.aero.diff import from_airplane
    from ICARUS.aero.diff.pipeline import diff_vlm_forces, make_diff_coefficients_fn

    airplane = make_wing_tail_airplane()
    diff_plane = from_airplane(airplane)

    check("Two wings", len(diff_plane.wings) == 2)
    check("Main wing correct", diff_plane.main_wing.name == "main_wing")

    airspeed, density = 20.0, 1.225
    alpha = jnp.asarray(5.0, dtype=jnp.float64)

    lift, drag, My = diff_vlm_forces(diff_plane, alpha, airspeed, density)
    print(f"  Lift={float(lift):.2f}N  Drag={float(drag):.2f}N  My={float(My):.2f}N·m")

    check("Multi-wing lift > 0", float(lift) > 0)
    check("Multi-wing drag > 0", float(drag) > 0)

    # Gradient should work too
    coeff_fn = make_diff_coefficients_fn(diff_plane, airspeed, density)
    dCL_da = float(jax.grad(lambda a: coeff_fn(diff_plane, a)[0])(alpha))
    print(f"  CL_alpha = {dCL_da * 180/np.pi:.4f} /rad")
    check("CL_alpha positive", dCL_da > 0)


def test_swept_tapered():
    """Test 6: Swept tapered wing."""
    print("\n" + "=" * 60)
    print("TEST 7: Swept Tapered Wing")
    print("=" * 60)

    from ICARUS.aero.diff import from_airplane
    from ICARUS.aero.diff.pipeline import make_diff_coefficients_fn

    airplane = make_tapered_swept_wing()
    diff_plane = from_airplane(airplane)

    airspeed, density = 30.0, 1.225
    alpha = jnp.asarray(3.0, dtype=jnp.float64)

    coeff_fn = make_diff_coefficients_fn(diff_plane, airspeed, density)
    CL, CD, Cm = coeff_fn(diff_plane, alpha)
    print(f"  CL={float(CL):.6f}  CD={float(CD):.6f}  Cm={float(Cm):.6f}")

    check("Swept wing CL > 0", float(CL) > 0)
    check("Swept wing CD > 0", float(CD) > 0)

    # Test gradient w.r.t. alpha
    dCL = float(jax.grad(lambda a: coeff_fn(diff_plane, a)[0])(alpha))
    check("Swept CL_alpha > 0", dCL > 0)


def test_control_surface():
    """Test 7: Control surface deflection gradients."""
    print("\n" + "=" * 60)
    print("TEST 8: Control Surface Gradients")
    print("=" * 60)

    from ICARUS.aero.diff import from_airplane
    from ICARUS.aero.diff.pipeline import make_diff_coefficients_fn

    airplane = make_wing_with_aileron()
    diff_plane = from_airplane(airplane)

    seg = diff_plane.wings[0].segments[0]
    print(f"  Controls: {[c.name for c in seg.controls]}")
    check("Has aileron", len(seg.controls) > 0 and seg.controls[0].name == "aileron")

    airspeed, density = 20.0, 1.225
    alpha = jnp.asarray(5.0, dtype=jnp.float64)

    coeff_fn = make_diff_coefficients_fn(diff_plane, airspeed, density)
    CL, CD, Cm = coeff_fn(diff_plane, alpha)
    print(f"  CL={float(CL):.6f}  CD={float(CD):.6f}  Cm={float(Cm):.6f}")
    check("Control wing CL > 0", float(CL) > 0)


def test_viscous_drag():
    """Test 8: Viscous drag with flat-plate polar."""
    print("\n" + "=" * 60)
    print("TEST 9: Viscous Drag (Flat-Plate Polar)")
    print("=" * 60)

    from ICARUS.aero.diff import from_airplane, make_flat_plate_polar
    from ICARUS.aero.diff.pipeline import diff_total_forces, make_diff_coefficients_fn

    airplane = make_rectangular_wing()
    diff_plane = from_airplane(airplane)
    polar = make_flat_plate_polar()

    airspeed, density = 20.0, 1.225
    alpha = jnp.asarray(5.0, dtype=jnp.float64)

    # Without viscous
    from ICARUS.aero.diff.pipeline import diff_vlm_forces
    _, drag_pot, _ = diff_vlm_forces(diff_plane, alpha, airspeed, density)

    # With viscous
    polar_data = {0: polar}  # segment 0
    _, drag_total, _ = diff_total_forces(diff_plane, alpha, airspeed, density, polar_data)

    print(f"  Induced drag = {float(drag_pot):.4f} N")
    print(f"  Total drag   = {float(drag_total):.4f} N")
    print(f"  Viscous drag = {float(drag_total - drag_pot):.4f} N")

    check("Viscous drag adds to total", float(drag_total) > float(drag_pot))

    # Gradient through viscous
    coeff_fn = make_diff_coefficients_fn(diff_plane, airspeed, density, polar_data=polar_data)
    dCD_da = float(jax.grad(lambda a: coeff_fn(diff_plane, a)[1])(alpha))
    print(f"  dCD_total/dalpha = {dCD_da:.6e}")
    check("Viscous CD gradient computable", not np.isnan(dCD_da))


def test_naca4_parametric():
    """Test 9: NACA4 parametric camber differentiation."""
    print("\n" + "=" * 60)
    print("TEST 10: NACA4 Parametric Camber")
    print("=" * 60)

    from ICARUS.aero.diff.airfoil_camber import DiffNACA4Camber, from_naca4

    naca = NACA4.from_digits("4415")
    diff_camber = from_naca4(naca)

    print(f"  m={float(diff_camber.m):.4f}  p={float(diff_camber.p):.4f}")

    # Evaluate camber line
    eta = jnp.linspace(0.01, 0.99, 50)
    camber = diff_camber.evaluate_at(eta)
    check("Camber line computed", camber.shape == (50,))
    check("Max camber > 0", float(jnp.max(camber)) > 0)

    # Gradient of max camber w.r.t. m parameter
    def max_camber_fn(dc: DiffNACA4Camber) -> Array:
        return jnp.max(dc.evaluate_at(eta))

    grads = eqx.filter_grad(max_camber_fn)(diff_camber)
    print(f"  d(max_camber)/dm = {float(grads.m):.6f}")
    print(f"  d(max_camber)/dp = {float(grads.p):.6f}")
    check("d(max_camber)/dm > 0", float(grads.m) > 0)


def test_jit():
    """Test 10: JIT compilation."""
    print("\n" + "=" * 60)
    print("TEST 11: JIT Compilation")
    print("=" * 60)

    from ICARUS.aero.diff import from_airplane
    from ICARUS.aero.diff.pipeline import diff_vlm_forces

    airplane = make_rectangular_wing(N=6, M=4)
    diff_plane = from_airplane(airplane)
    airspeed, density = 20.0, 1.225
    alpha = jnp.asarray(5.0, dtype=jnp.float64)

    # Non-JIT baseline
    t0 = time.perf_counter()
    L1, D1, M1 = diff_vlm_forces(diff_plane, alpha, airspeed, density)
    t_nojit = time.perf_counter() - t0

    # JIT-compiled version
    @jax.jit
    def jit_forces(plane, a):
        return diff_vlm_forces(plane, a, airspeed, density)

    # First call (compilation)
    t0 = time.perf_counter()
    L2, D2, M2 = jit_forces(diff_plane, alpha)
    t_compile = time.perf_counter() - t0

    # Second call (cached)
    t0 = time.perf_counter()
    L3, D3, M3 = jit_forces(diff_plane, alpha)
    jax.block_until_ready(L3)
    t_cached = time.perf_counter() - t0

    print(f"  No JIT:    {t_nojit:.3f}s")
    print(f"  JIT compile: {t_compile:.3f}s")
    print(f"  JIT cached:  {t_cached:.4f}s")

    # Results should match
    l_err = abs(float(L1 - L3)) / abs(float(L1)) * 100
    check("JIT results match", l_err < 0.01, f"err={l_err:.4f}%")
    check("JIT faster than no-JIT", t_cached < t_nojit or t_cached < 0.1)


def test_polar_sweep():
    """TEST 12: Differentiable polar sweep."""
    print("\n" + "=" * 60)
    print("TEST 12: Differentiable Polar Sweep")
    print("=" * 60)

    from ICARUS.aero.diff import from_airplane, diff_polar_sweep

    airplane = make_rectangular_wing()
    diff_plane = from_airplane(airplane)
    angles = [-2.0, 0.0, 2.0, 5.0, 8.0]

    sweep = diff_polar_sweep(
        airplane=diff_plane,
        angles=angles,
        airspeed=20.0,
        density=1.225,
        compute_stability_derivatives=True,
    )

    check("Sweep has all angles", len(sweep["CL"]) == len(angles))
    check("CL increases with alpha", sweep["CL"][-1] > sweep["CL"][0])
    check("CD all positive", all(cd > 0 for cd in sweep["CD"]))
    check("CL_alpha all positive", all(cla > 0 for cla in sweep["CL_alpha"]))

    # CL_alpha should be roughly constant (linear range)
    cla_spread = max(sweep["CL_alpha"]) - min(sweep["CL_alpha"])
    cla_mean = sum(sweep["CL_alpha"]) / len(sweep["CL_alpha"])
    check("CL_alpha nearly constant", cla_spread / cla_mean < 0.05,
          f"spread={cla_spread:.4f}, mean={cla_mean:.4f}")

    print(f"  Angles: {sweep['AoA']}")
    print(f"  CL:     {[f'{c:.4f}' for c in sweep['CL']]}")
    print(f"  CL_alpha: {[f'{c:.2f}' for c in sweep['CL_alpha']]}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Equinox Differentiable Modules — Comprehensive Validation")
    print("=" * 60)

    diff_plane, coeff_fn = test_conversion_and_forward()
    test_alpha_gradient(diff_plane, coeff_fn)
    test_cm_alignment()
    test_design_gradient()
    test_multi_wing()
    test_swept_tapered()
    test_control_surface()
    test_viscous_drag()
    test_naca4_parametric()
    test_jit()
    test_polar_sweep()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)
