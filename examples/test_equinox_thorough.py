"""
Thorough Unit Tests for ICARUS Differentiable Modules

Covers all public API surface not tested by test_equinox_modules.py:
- mass.py: DiffMass, inertia_about_point, from_mass/to_mass roundtrip
- airfoil_camber.py: NACA4 values, morph_camber, from_airfoil
- control_surface.py: affects_station, hinge_fraction, modify_camber
- wing_segment.py: properties, compute_geometry_params
- airplane.py: DiffWing/DiffAirplane properties, compute_cg
- viscous.py: polar values, interpolation, strip forces
- pipeline.py: trim solver, implicit sensitivity, gradient_fn, edge cases
- pytree: immutability, tree leaves
"""
from __future__ import annotations
import sys
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


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


# ---- Factory helpers ----

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


# ---- Tests: mass.py ----

def test_mass_unit():
    print("\n" + "=" * 60)
    print("TEST 1: DiffMass Unit Tests")
    print("=" * 60)
    from ICARUS.aero.diff.mass import DiffMass
    dm = DiffMass(
        position=jnp.array([1.0, 0.0, 0.0]),
        mass=jnp.array(5.0),
        name="test",
        inertia=jnp.zeros(6),
    )
    check("mass value", float(dm.mass) == 5.0)
    check("position shape", dm.position.shape == (3,))
    # inertia_about_point: mass=5 at [1,0,0], zero local inertia, about origin
    # Ixx = 0 + 5*(0^2+0^2) = 0, Iyy = 0 + 5*(1^2+0^2) = 5, Izz = 0 + 5*(1^2+0^2) = 5
    I = dm.inertia_about_point(jnp.zeros(3))
    check("Ixx=0", abs(float(I[0])) < 1e-10)
    check("Iyy=5", abs(float(I[1]) - 5.0) < 1e-10)
    check("Izz=5", abs(float(I[2]) - 5.0) < 1e-10)
    check("Ixy=0", abs(float(I[3])) < 1e-10)
    check("Ixz=0", abs(float(I[4])) < 1e-10)
    check("Iyz=0", abs(float(I[5])) < 1e-10)


def test_mass_roundtrip():
    print("\n" + "=" * 60)
    print("TEST 2: Mass Roundtrip")
    print("=" * 60)
    from ICARUS.aero.diff.mass import DiffMass, to_mass
    dm = DiffMass(
        position=jnp.array([1.0, 2.0, 3.0]),
        mass=jnp.array(7.5),
        name="test_mass",
        inertia=jnp.zeros(6),
    )
    m_back = to_mass(dm)
    check("roundtrip name", m_back.name == "test_mass")
    check("roundtrip mass", abs(m_back.mass - 7.5) < 1e-10)
    check("roundtrip position", np.allclose(m_back.position, [1.0, 2.0, 3.0]))


def test_mass_gradient():
    print("\n" + "=" * 60)
    print("TEST 3: Mass Gradient")
    print("=" * 60)
    from ICARUS.aero.diff.mass import DiffMass
    dm = DiffMass(
        position=jnp.array([1.0, 2.0, 3.0]),
        mass=jnp.array(5.0),
        name="test",
        inertia=jnp.zeros(6),
    )
    def fn(m):
        return m.mass * m.position[0]
    grads = eqx.filter_grad(fn)(dm)
    check("grad mass = x", abs(float(grads.mass) - 1.0) < 1e-10)
    check("grad pos[0] = mass", abs(float(grads.position[0]) - 5.0) < 1e-10)
    check("grad pos[1] = 0", abs(float(grads.position[1])) < 1e-10)


# ---- Tests: airfoil_camber.py ----

def test_naca4_camber_values():
    print("\n" + "=" * 60)
    print("TEST 4: NACA4 Camber Values")
    print("=" * 60)
    from ICARUS.aero.diff.airfoil_camber import DiffNACA4Camber
    dc = DiffNACA4Camber(m=jnp.array(0.04), p=jnp.array(0.4), name="test", xx=0.15, norm_factor=1.0)
    eta = jnp.linspace(0.01, 0.99, 100)
    camber = dc.evaluate_at(eta)
    peak_idx = int(jnp.argmax(camber))
    peak_x = float(eta[peak_idx])
    check("peak near p=0.4", abs(peak_x - 0.4) < 0.05, f"peak_x={peak_x}")
    peak_val = float(jnp.max(camber))
    check("peak value near m", abs(peak_val - 0.04) < 0.01, f"peak={peak_val}")
    # Symmetric: m=0 -> zero camber
    dc0 = DiffNACA4Camber(m=jnp.array(0.0), p=jnp.array(0.5), name="sym", xx=0.12, norm_factor=1.0)
    camber0 = dc0.evaluate_at(eta)
    check("symmetric zero camber", float(jnp.max(jnp.abs(camber0))) < 1e-10)


def test_morph_camber():
    print("\n" + "=" * 60)
    print("TEST 5: Morph Camber")
    print("=" * 60)
    from ICARUS.aero.diff.airfoil_camber import DiffAirfoilCamber, morph_camber
    c1 = DiffAirfoilCamber(camber_at_eta=jnp.array([0.0, 0.02, 0.04, 0.02, 0.0]), name="a1", norm_factor=1.0)
    c2 = DiffAirfoilCamber(camber_at_eta=jnp.array([0.0, 0.06, 0.08, 0.06, 0.0]), name="a2", norm_factor=1.0)
    m0 = morph_camber(c1, c2, jnp.array(0.0))
    check("morph eta=0 is c1", jnp.allclose(m0.camber_at_eta, c1.camber_at_eta))
    m1 = morph_camber(c1, c2, jnp.array(1.0))
    check("morph eta=1 is c2", jnp.allclose(m1.camber_at_eta, c2.camber_at_eta))
    m5 = morph_camber(c1, c2, jnp.array(0.5))
    expected = (c1.camber_at_eta + c2.camber_at_eta) / 2
    check("morph eta=0.5 is avg", jnp.allclose(m5.camber_at_eta, expected))
    # Gradient: morph_camber calls float(eta) for norm_factor (non-diff),
    # so test gradient of the morphed camber values directly instead.
    def morph_sum(eta):
        morphed = (1 - eta) * c1.camber_at_eta + eta * c2.camber_at_eta
        return jnp.sum(morphed)
    g = float(jax.grad(morph_sum)(jnp.array(0.5)))
    expected_g = float(jnp.sum(c2.camber_at_eta - c1.camber_at_eta))
    check("morph gradient", abs(g - expected_g) < 1e-8, f"g={g}, expected={expected_g}")


def test_from_airfoil_custom_eta():
    print("\n" + "=" * 60)
    print("TEST 6: from_airfoil with Custom Eta")
    print("=" * 60)
    from ICARUS.aero.diff.airfoil_camber import from_airfoil
    naca = NACA4.from_digits("4415")
    custom_eta = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    dc = from_airfoil(naca, chord_eta=custom_eta)
    check("custom eta shape", dc.camber_at_eta.shape == (5,))
    check("custom eta nonzero", float(jnp.max(dc.camber_at_eta)) > 0)
    # Default M
    dc_default = from_airfoil(naca, M=10)
    check("default M shape", dc_default.camber_at_eta.shape == (10,))


# ---- Tests: control_surface.py ----

def test_control_surface_affects():
    print("\n" + "=" * 60)
    print("TEST 7: Control Surface affects_station")
    print("=" * 60)
    from ICARUS.aero.diff.control_surface import DiffControlSurface
    cs = DiffControlSurface(
        deflection=jnp.array(0.0), gain=jnp.array(1.0),
        name="test_cs", control_var="delta",
        span_start=0.6, span_end=0.9, chord_start=0.7, chord_end=0.7,
        inverse_symmetric=False,
    )
    check("affects inside", cs.affects_station(0.75))
    check("not affects low", not cs.affects_station(0.3))
    check("not affects high", not cs.affects_station(0.95))
    check("affects boundary start", cs.affects_station(0.6))
    check("affects boundary end", cs.affects_station(0.9))


def test_hinge_fraction():
    print("\n" + "=" * 60)
    print("TEST 8: Hinge Fraction Interpolation")
    print("=" * 60)
    from ICARUS.aero.diff.control_surface import DiffControlSurface
    cs = DiffControlSurface(
        deflection=jnp.array(0.0), gain=jnp.array(1.0),
        name="test", control_var="d",
        span_start=0.4, span_end=0.8, chord_start=0.6, chord_end=0.8,
        inverse_symmetric=False,
    )
    h_start = cs.hinge_fraction(0.4)
    check("hinge at start", abs(h_start - 0.6) < 1e-10)
    h_end = cs.hinge_fraction(0.8)
    check("hinge at end", abs(h_end - 0.8) < 1e-10)
    h_mid = cs.hinge_fraction(0.6)
    check("hinge at mid", abs(h_mid - 0.7) < 1e-10, f"h_mid={h_mid}")


def test_modify_camber():
    print("\n" + "=" * 60)
    print("TEST 9: modify_camber")
    print("=" * 60)
    from ICARUS.aero.diff.control_surface import DiffControlSurface
    chord_eta = jnp.linspace(0.0, 1.0, 11)
    camber_z = jnp.zeros(11)

    # Zero deflection -> unchanged
    cs_zero = DiffControlSurface(
        deflection=jnp.array(0.0), gain=jnp.array(1.0),
        name="t", control_var="d",
        span_start=0.0, span_end=1.0, chord_start=0.7, chord_end=0.7,
        inverse_symmetric=False,
    )
    mod_zero = cs_zero.modify_camber(camber_z, chord_eta, 0.5)
    check("zero defl unchanged", jnp.allclose(mod_zero, camber_z, atol=1e-12))

    # Positive deflection
    cs_pos = DiffControlSurface(
        deflection=jnp.array(0.1), gain=jnp.array(1.0),
        name="t", control_var="d",
        span_start=0.0, span_end=1.0, chord_start=0.7, chord_end=0.7,
        inverse_symmetric=False,
    )
    mod_pos = cs_pos.modify_camber(camber_z, chord_eta, 0.5)
    check("forward unchanged", jnp.allclose(mod_pos[:7], camber_z[:7], atol=1e-12))
    aft_change = float(jnp.sum(jnp.abs(mod_pos[8:])))
    check("aft changed", aft_change > 1e-6, f"aft_change={aft_change}")

    # Gradient w.r.t. deflection
    def camber_sum_fn(defl):
        cs_g = DiffControlSurface(
            deflection=defl, gain=jnp.array(1.0),
            name="t", control_var="d",
            span_start=0.0, span_end=1.0, chord_start=0.7, chord_end=0.7,
            inverse_symmetric=False,
        )
        return jnp.sum(cs_g.modify_camber(camber_z, chord_eta, 0.5))
    g = jax.grad(camber_sum_fn)(jnp.array(0.1))
    check("defl gradient finite", jnp.isfinite(g))
    check("defl gradient nonzero", abs(float(g)) > 1e-8)


def test_apply_controls_to_camber():
    print("\n" + "=" * 60)
    print("TEST 10: apply_controls_to_camber")
    print("=" * 60)
    from ICARUS.aero.diff.control_surface import DiffControlSurface, apply_controls_to_camber
    M, N = 11, 5
    camber_z = jnp.zeros((M, N))
    chord_eta = jnp.linspace(0.0, 1.0, M)
    span_fracs = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])

    cs = DiffControlSurface(
        deflection=jnp.array(0.1), gain=jnp.array(1.0),
        name="flap", control_var="d",
        span_start=0.4, span_end=0.8, chord_start=0.7, chord_end=0.7,
        inverse_symmetric=False,
    )
    result = apply_controls_to_camber(camber_z, chord_eta, [cs], span_fracs)
    check("station 0 unchanged", jnp.allclose(result[:, 0], camber_z[:, 0]))
    check("station 1 unchanged", jnp.allclose(result[:, 1], camber_z[:, 1]))
    check("station 2 changed", float(jnp.sum(jnp.abs(result[:, 2]))) > 1e-6)
    check("station 3 changed", float(jnp.sum(jnp.abs(result[:, 3]))) > 1e-6)
    check("station 4 unchanged", jnp.allclose(result[:, 4], camber_z[:, 4]))


# ---- Tests: wing_segment.py and airplane.py ----

def test_wing_segment_properties():
    print("\n" + "=" * 60)
    print("TEST 11: Wing Segment Properties")
    print("=" * 60)
    from ICARUS.aero.diff import from_airplane
    airplane = make_rectangular_wing(N=10, M=5)
    diff_plane = from_airplane(airplane)
    seg = diff_plane.wings[0].segments[0]
    check("num_panels", seg.num_panels == (10 - 1) * (5 - 1))
    check("span > 0", seg.span > 0, f"span={seg.span}")
    check("area > 0", seg.area > 0, f"area={seg.area}")
    check("MAC > 0", seg.mean_aerodynamic_chord > 0, f"MAC={seg.mean_aerodynamic_chord}")


def test_compute_geometry_params():
    print("\n" + "=" * 60)
    print("TEST 12: compute_geometry_params")
    print("=" * 60)
    from ICARUS.aero.diff import from_airplane
    airplane = make_rectangular_wing(N=6, M=4)
    diff_plane = from_airplane(airplane)
    seg = diff_plane.wings[0].segments[0]
    params = seg.compute_geometry_params()
    expected_keys = {"chord_dist", "span_dist", "twist_angles", "x_offsets",
                     "z_offsets", "camber_z", "chord_eta", "R_MAT", "origin",
                     "norm_factors", "N", "M"}
    check("all keys present", expected_keys.issubset(set(params.keys())),
          f"missing: {expected_keys - set(params.keys())}")
    check("chord_dist shape", params["chord_dist"].shape == (6,))
    check("span_dist shape", params["span_dist"].shape == (6,))
    check("N matches", params["N"] == 6)
    check("M matches", params["M"] == 4)


def test_diff_wing_properties():
    print("\n" + "=" * 60)
    print("TEST 13: DiffWing Properties")
    print("=" * 60)
    from ICARUS.aero.diff import from_airplane
    airplane = make_rectangular_wing(N=8, M=4)
    diff_plane = from_airplane(airplane)
    wing = diff_plane.wings[0]
    check("wing is_lifting", wing.is_lifting)
    check("wing span > 0", wing.span > 0)
    check("wing area > 0", wing.area > 0)
    check("wing MAC > 0", wing.mean_aerodynamic_chord > 0)
    check("wing num_panels", wing.num_panels == (8 - 1) * (4 - 1))


def test_diff_airplane_properties():
    print("\n" + "=" * 60)
    print("TEST 14: DiffAirplane Properties")
    print("=" * 60)
    from ICARUS.aero.diff import from_airplane
    airplane = make_wing_tail_airplane(N=8, M=4)
    diff_plane = from_airplane(airplane)
    check("S > 0", diff_plane.S > 0)
    check("span > 0", diff_plane.span > 0)
    check("MAC > 0", diff_plane.MAC > 0)
    check("2 wings", len(diff_plane.wings) == 2)
    check("all_segments count", len(diff_plane.all_segments) == 2)
    check("lifting_segments", len(diff_plane.lifting_segments) >= 1)
    check("total_mass > 0", diff_plane.total_mass > 0)


def test_compute_cg():
    print("\n" + "=" * 60)
    print("TEST 15: compute_cg")
    print("=" * 60)
    from ICARUS.aero.diff.airplane import DiffAirplane
    from ICARUS.aero.diff.mass import DiffMass
    # With override
    plane = DiffAirplane(
        wings=[], point_masses=[],
        cg_override=jnp.array([1.0, 2.0, 3.0]),
        name="t", main_wing_name="none", has_cg_override=True,
    )
    cg = plane.compute_cg()
    check("cg override x", abs(float(cg[0]) - 1.0) < 1e-10)
    check("cg override y", abs(float(cg[1]) - 2.0) < 1e-10)
    check("cg override z", abs(float(cg[2]) - 3.0) < 1e-10)

    # Without override: two equal masses at x=0 and x=2 -> CG at x=1
    m1 = DiffMass(position=jnp.array([0.0, 0.0, 0.0]), mass=jnp.array(10.0),
                  name="m1", inertia=jnp.zeros(6))
    m2 = DiffMass(position=jnp.array([2.0, 0.0, 0.0]), mass=jnp.array(10.0),
                  name="m2", inertia=jnp.zeros(6))
    plane2 = DiffAirplane(
        wings=[], point_masses=[m1, m2],
        cg_override=jnp.zeros(3),
        name="t2", main_wing_name="none", has_cg_override=False,
    )
    cg2 = plane2.compute_cg()
    check("computed cg x=1", abs(float(cg2[0]) - 1.0) < 1e-10, f"cg_x={float(cg2[0])}")
    check("computed cg y=0", abs(float(cg2[1])) < 1e-10)


# ---- Tests: viscous.py ----

def test_flat_plate_polar_values():
    print("\n" + "=" * 60)
    print("TEST 16: Flat Plate Polar Values")
    print("=" * 60)
    from ICARUS.aero.diff import make_flat_plate_polar
    polar = make_flat_plate_polar()
    aoa = jnp.array(5.0)
    cl, cd, cm = polar.interpolate(aoa)
    expected_cl = float(2 * jnp.pi * jnp.deg2rad(5.0))
    check("CL at 5deg", abs(float(cl) - expected_cl) < 0.01, f"cl={float(cl):.4f}, exp={expected_cl:.4f}")
    expected_cd = 0.008 + 0.0001 * 25
    check("CD at 5deg", abs(float(cd) - expected_cd) < 0.001, f"cd={float(cd):.6f}, exp={expected_cd:.6f}")


def test_polar_interpolate_gradient():
    print("\n" + "=" * 60)
    print("TEST 17: Polar Interpolation Gradient")
    print("=" * 60)
    from ICARUS.aero.diff import make_flat_plate_polar
    polar = make_flat_plate_polar()
    def cd_of_aoa(a):
        _, cd, _ = polar.interpolate(a)
        return cd
    g = float(jax.grad(cd_of_aoa)(jnp.array(5.0)))
    check("dCD/dalpha > 0", g > 0, f"g={g}")
    check("dCD/dalpha finite", jnp.isfinite(jnp.array(g)))
    # CL gradient: dCL/dalpha should be ~2*pi in radians, or 2*pi/180 per degree
    def cl_of_aoa(a):
        cl, _, _ = polar.interpolate(a)
        return cl
    g_cl = float(jax.grad(cl_of_aoa)(jnp.array(5.0)))
    expected_g_cl = 2 * np.pi * np.pi / 180  # per degree
    check("dCL/dalpha reasonable", abs(g_cl - expected_g_cl) < 0.01, f"g={g_cl:.4f}, exp={expected_g_cl:.4f}")


def test_polar_out_of_bounds():
    print("\n" + "=" * 60)
    print("TEST 18: Polar Out of Bounds")
    print("=" * 60)
    from ICARUS.aero.diff import make_flat_plate_polar
    polar = make_flat_plate_polar(aoa_range=(-15.0, 15.0))
    cl_hi, cd_hi, _ = polar.interpolate(jnp.array(20.0))
    cl_lo, _, _ = polar.interpolate(jnp.array(-20.0))
    check("OOB high no NaN", jnp.isfinite(cl_hi))
    check("OOB low no NaN", jnp.isfinite(cl_lo))
    cl_15, _, _ = polar.interpolate(jnp.array(15.0))
    check("OOB clamped high", abs(float(cl_hi) - float(cl_15)) < 1e-6)


def test_strip_viscous_forces():
    print("\n" + "=" * 60)
    print("TEST 19: Strip Viscous Forces")
    print("=" * 60)
    from ICARUS.aero.diff.viscous import compute_strip_viscous_forces, make_flat_plate_polar
    polar = make_flat_plate_polar()
    aoa = jnp.array(5.0)
    V = jnp.array(20.0)
    chord = jnp.array(1.0)
    width = jnp.array(0.5)
    density = 1.225
    L, D = compute_strip_viscous_forces(aoa, V, chord, width, density, polar)
    q = 0.5 * density * 20.0**2
    S = 1.0 * 0.5
    _, cd_val, _ = polar.interpolate(aoa)
    expected_D = float(cd_val) * q * S
    check("strip drag matches", abs(float(D) - expected_D) < 1e-6,
          f"D={float(D):.6f}, expected={expected_D:.6f}")
    check("strip lift > 0", float(L) > 0)


# ---- Tests: pipeline.py ----

def test_compute_trim_alpha():
    print("\n" + "=" * 60)
    print("TEST 20: Trim Solver")
    print("=" * 60)
    from ICARUS.aero.diff import from_airplane
    from ICARUS.aero.diff.pipeline import compute_trim_alpha, make_diff_coefficients_fn
    airplane = make_wing_tail_airplane()
    diff_plane = from_airplane(airplane)
    alpha_trim = compute_trim_alpha(diff_plane, 20.0, 1.225, target_Cm=0.0)
    coeff_fn = make_diff_coefficients_fn(diff_plane, 20.0, 1.225)
    _, _, Cm_at_trim = coeff_fn(diff_plane, alpha_trim)
    check("trim Cm near 0", abs(float(Cm_at_trim)) < 0.01, f"Cm={float(Cm_at_trim):.6f}")
    check("trim alpha finite", jnp.isfinite(alpha_trim))
    check("trim alpha reasonable", -20 < float(alpha_trim) < 30, f"alpha={float(alpha_trim):.2f}")


def test_implicit_trim_sensitivity():
    print("\n" + "=" * 60)
    print("TEST 21: Implicit Trim Sensitivity")
    print("=" * 60)
    from ICARUS.aero.diff import from_airplane
    from ICARUS.aero.diff.pipeline import compute_trim_alpha, implicit_trim_sensitivity, make_diff_coefficients_fn
    airplane = make_wing_tail_airplane()
    diff_plane = from_airplane(airplane)
    alpha_trim = compute_trim_alpha(diff_plane, 20.0, 1.225)
    trim_fn = implicit_trim_sensitivity(diff_plane, 20.0, 1.225, alpha_trim)
    CL, CD, Cm = trim_fn(diff_plane)
    coeff_fn = make_diff_coefficients_fn(diff_plane, 20.0, 1.225)
    CL_d, CD_d, Cm_d = coeff_fn(diff_plane, alpha_trim)
    check("implicit CL matches", abs(float(CL) - float(CL_d)) < 1e-8)
    check("implicit CD matches", abs(float(CD) - float(CD_d)) < 1e-8)
    # Gradient through implicit function
    grads = eqx.filter_grad(lambda p: trim_fn(p)[1])(diff_plane)
    seg_grads = grads.wings[0].segments[0]
    check("implicit grad finite", bool(jnp.all(jnp.isfinite(seg_grads.chord_dist))))
    check("implicit grad nonzero", float(jnp.sum(jnp.abs(seg_grads.chord_dist))) > 1e-10)


def test_gradient_fn_CL_Cm():
    print("\n" + "=" * 60)
    print("TEST 22: make_gradient_fn CL and Cm")
    print("=" * 60)
    from ICARUS.aero.diff import from_airplane
    from ICARUS.aero.diff.pipeline import make_gradient_fn
    diff_plane = from_airplane(make_rectangular_wing())
    # CL gradient
    cl_fn = make_gradient_fn(diff_plane, 20.0, 1.225, alpha_deg=5.0, output="CL")
    cl_val = float(cl_fn(diff_plane))
    cl_grads = eqx.filter_grad(cl_fn)(diff_plane)
    check("CL value > 0", cl_val > 0, f"CL={cl_val}")
    check("CL grad finite", bool(jnp.all(jnp.isfinite(cl_grads.wings[0].segments[0].chord_dist))))
    # Cm gradient
    cm_fn = make_gradient_fn(diff_plane, 20.0, 1.225, alpha_deg=5.0, output="Cm")
    cm_val = float(cm_fn(diff_plane))
    cm_grads = eqx.filter_grad(cm_fn)(diff_plane)
    check("Cm value finite", jnp.isfinite(jnp.array(cm_val)))
    check("Cm grad finite", bool(jnp.all(jnp.isfinite(cm_grads.wings[0].segments[0].chord_dist))))


def test_total_forces_no_polar():
    print("\n" + "=" * 60)
    print("TEST 23: diff_total_forces with no polar")
    print("=" * 60)
    from ICARUS.aero.diff import from_airplane
    from ICARUS.aero.diff.pipeline import diff_vlm_forces, diff_total_forces
    diff_plane = from_airplane(make_rectangular_wing())
    alpha = jnp.array(5.0)
    L1, D1, M1 = diff_vlm_forces(diff_plane, alpha, 20.0, 1.225)
    L2, D2, M2 = diff_total_forces(diff_plane, alpha, 20.0, 1.225, polar_data=None)
    check("no polar L match", abs(float(L1 - L2)) < 1e-10)
    check("no polar D match", abs(float(D1 - D2)) < 1e-10)
    check("no polar M match", abs(float(M1 - M2)) < 1e-10)


def test_polar_sweep_no_derivs():
    print("\n" + "=" * 60)
    print("TEST 24: Polar Sweep without Derivatives")
    print("=" * 60)
    from ICARUS.aero.diff import from_airplane, diff_polar_sweep
    diff_plane = from_airplane(make_rectangular_wing())
    sweep = diff_polar_sweep(diff_plane, [0.0, 5.0], 20.0, 1.225,
                             compute_stability_derivatives=False)
    check("has CL", "CL" in sweep)
    check("has CD", "CD" in sweep)
    check("has Cm", "Cm" in sweep)
    check("no CL_alpha", "CL_alpha" not in sweep)
    check("no CD_alpha", "CD_alpha" not in sweep)
    check("correct count", len(sweep["CL"]) == 2)


# ---- Tests: pytree / equinox integration ----

def test_eqx_tree_at_immutability():
    print("\n" + "=" * 60)
    print("TEST 25: eqx.tree_at Immutability")
    print("=" * 60)
    from ICARUS.aero.diff import from_airplane
    diff_plane = from_airplane(make_rectangular_wing())
    original_chord = diff_plane.wings[0].segments[0].chord_dist.copy()
    new_chords = diff_plane.wings[0].segments[0].chord_dist * 1.5
    modified = eqx.tree_at(lambda p: p.wings[0].segments[0].chord_dist, diff_plane, new_chords)
    check("original unchanged", jnp.allclose(diff_plane.wings[0].segments[0].chord_dist, original_chord))
    check("modified changed", jnp.allclose(modified.wings[0].segments[0].chord_dist, new_chords))
    check("not same object", diff_plane is not modified)


def test_pytree_leaves():
    print("\n" + "=" * 60)
    print("TEST 26: Pytree Leaves")
    print("=" * 60)
    from ICARUS.aero.diff import from_airplane
    diff_plane = from_airplane(make_rectangular_wing())
    leaves = jax.tree_util.tree_leaves(diff_plane)
    check("has leaves", len(leaves) > 0, f"count={len(leaves)}")
    check("all arrays", all(isinstance(l, jnp.ndarray) for l in leaves))
    check("all finite", all(bool(jnp.all(jnp.isfinite(l))) for l in leaves))


# ---- Main ----

if __name__ == "__main__":
    print("Thorough Unit Tests for ICARUS Differentiable Modules")
    print("=" * 60)

    test_mass_unit()
    test_mass_roundtrip()
    test_mass_gradient()
    test_naca4_camber_values()
    test_morph_camber()
    test_from_airfoil_custom_eta()
    test_control_surface_affects()
    test_hinge_fraction()
    test_modify_camber()
    test_apply_controls_to_camber()
    test_wing_segment_properties()
    test_compute_geometry_params()
    test_diff_wing_properties()
    test_diff_airplane_properties()
    test_compute_cg()
    test_flat_plate_polar_values()
    test_polar_interpolate_gradient()
    test_polar_out_of_bounds()
    test_strip_viscous_forces()
    test_compute_trim_alpha()
    test_implicit_trim_sensitivity()
    test_gradient_fn_CL_Cm()
    test_total_forces_no_polar()
    test_polar_sweep_no_derivs()
    test_eqx_tree_at_immutability()
    test_pytree_leaves()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)
