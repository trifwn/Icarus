"""
Differentiable Aerodynamic Modules (Equinox-based)

This package provides JAX-differentiable replacements for the core ICARUS
vehicle and aerodynamic classes using Equinox Modules. Each class is a
frozen pytree node, making it compatible with jax.grad, jax.jacobian,
jax.jit, and jax.vmap.

Architecture:
    Existing ICARUS classes (Mass, Airfoil, WingSurface, Airplane) are
    mutable, NumPy-based, and designed for imperative workflows. The Diff*
    modules mirror their structure but are:
    - Immutable (frozen Equinox modules)
    - JAX-native (all arrays are jax.numpy arrays)
    - Differentiable (dynamic fields are JAX-traced, static fields are not)

    Conversion functions (from_mass, from_airfoil, etc.) bridge between
    the existing classes and the differentiable modules.

Usage:
    >>> from ICARUS.aero.diff import DiffWingSegment, DiffAirplane
    >>> from ICARUS.aero.diff import from_airplane
    >>> from ICARUS.aero.diff.pipeline import make_gradient_fn
    >>>
    >>> # Convert existing airplane to differentiable form
    >>> diff_plane = from_airplane(airplane)
    >>>
    >>> # Create gradient function
    >>> grad_fn = make_gradient_fn(diff_plane, airspeed=20.0, density=1.225)
    >>> grads = jax.grad(grad_fn)(diff_plane)
"""

from .mass import DiffMass
from .mass import from_mass
from .airfoil_camber import DiffAirfoilCamber
from .airfoil_camber import from_airfoil
from .control_surface import DiffControlSurface
from .control_surface import from_control_surface
from .wing_segment import DiffWingSegment
from .wing_segment import from_wing_surface
from .airplane import DiffWing
from .airplane import DiffAirplane
from .airplane import from_airplane

from .airfoil_camber import DiffNACA4Camber
from .airfoil_camber import from_naca4
from .viscous import DiffPolarData
from .viscous import make_flat_plate_polar
from .viscous import load_polar_data
from .pipeline import diff_vlm_forces
from .pipeline import diff_total_forces
from .pipeline import make_diff_coefficients_fn
from .pipeline import make_gradient_fn
from .pipeline import compute_trim_alpha

__all__ = [
    "DiffMass",
    "from_mass",
    "DiffAirfoilCamber",
    "DiffNACA4Camber",
    "from_airfoil",
    "from_naca4",
    "DiffControlSurface",
    "from_control_surface",
    "DiffWingSegment",
    "from_wing_surface",
    "DiffWing",
    "DiffAirplane",
    "from_airplane",
    "DiffPolarData",
    "make_flat_plate_polar",
    "load_polar_data",
    "diff_vlm_forces",
    "diff_total_forces",
    "make_diff_coefficients_fn",
    "make_gradient_fn",
    "compute_trim_alpha",
]
