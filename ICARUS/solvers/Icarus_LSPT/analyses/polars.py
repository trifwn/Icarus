from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pandas as pd
from numpy import ndarray

from ICARUS.aero import LSPT_Plane
from ICARUS.aero.aerodynamic_results import AerodynamicResults
from ICARUS.database import Database

if TYPE_CHECKING:
    from ICARUS.core.types import FloatArray
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane

from ICARUS.aero.vlm import run_vlm_polar_analysis


def lspt_polars(
    plane: Airplane,
    state: State,
    angles: FloatArray | list[float],
    solver_parameters: dict[str, Any],
) -> pd.DataFrame:
    """Function to run the wing LLT solver

    Args:
        plane (Airplane): Airplane Object
        options (dict[str, Any]): Options
        solver_parameters (dict[str, Any]): Solver Options

    """
    DB = Database.get_instance()
    LSPTDIR = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="LSPT",
    )

    os.makedirs(LSPTDIR, exist_ok=True)
    # Generate the wing LLT solver

    lspt_plane = LSPT_Plane(
        plane=plane,
    )

    # Run the solver
    if not isinstance(angles, ndarray):
        angles = np.array(angles)

    results: AerodynamicResults = run_vlm_polar_analysis(
        plane=lspt_plane,
        state=state,
        angles=angles,
    )

    # Convert the results to a DataFrame
    df = results.to_polars_dataframe()

    # Save the results
    save_results(plane, state, df)
    return df


def lspt_polars_with_gradients(
    plane: Airplane,
    state: State,
    angles: FloatArray | list[float],
    solver_parameters: dict[str, Any],
    compute_design_gradients: bool = False,
    gradient_outputs: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Run LSPT polar sweep with JAX autodiff stability derivatives.

    This uses the Equinox differentiable pipeline to compute exact
    analytic derivatives alongside the polar sweep. The standard
    VLM results are also computed for comparison/storage.

    Args:
        plane: Airplane Object
        state: Flight state (provides airspeed, density)
        angles: Angles of attack (degrees)
        solver_parameters: Solver options
        compute_design_gradients: If True, also compute dCL/d(design),
            dCD/d(design), dCm/d(design) at each angle. Returns per-strip
            chord and twist gradients.
        gradient_outputs: Which coefficients to differentiate for design
            gradients. Default: ["CD"]. Options: "CL", "CD", "Cm".

    Returns:
        Tuple of (forces_df, gradients_df).
        forces_df: Standard polar results with added stability derivative columns.
        gradients_df: Design gradient DataFrame (or None if not requested).
    """
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    from ICARUS.aero.diff import (
        from_airplane,
        diff_polar_sweep,
        make_gradient_fn,
    )

    if not isinstance(angles, ndarray):
        angles = np.array(angles)

    # --- Standard polar sweep (existing solver, for storage/comparison) ---
    lspt_plane = LSPT_Plane(plane=plane)
    results: AerodynamicResults = run_vlm_polar_analysis(
        plane=lspt_plane, state=state, angles=angles,
    )
    forces_df = results.to_polars_dataframe()

    # --- Differentiable polar sweep ---
    diff_plane = from_airplane(plane)
    airspeed = float(state.velocity)
    density = float(state.environment.density)

    sweep = diff_polar_sweep(
        airplane=diff_plane,
        angles=angles.tolist(),
        airspeed=airspeed,
        density=density,
        compute_stability_derivatives=True,
    )

    # Add stability derivatives to forces DataFrame
    forces_df["CL_alpha"] = sweep["CL_alpha"]
    forces_df["CD_alpha"] = sweep["CD_alpha"]
    forces_df["Cm_alpha"] = sweep["Cm_alpha"]

    save_results(plane, state, forces_df)

    # --- Design gradients (optional, more expensive) ---
    gradients_df = None
    if compute_design_gradients:
        if gradient_outputs is None:
            gradient_outputs = ["CD"]

        grad_rows = []
        for angle in angles:
            row = {"AoA": float(angle)}

            for output in gradient_outputs:
                grad_fn = make_gradient_fn(
                    diff_plane, airspeed, density,
                    alpha_deg=float(angle), output=output,
                )
                grads = eqx.filter_grad(grad_fn)(diff_plane)

                # Extract per-segment chord and twist gradients
                for wing in grads.wings:
                    for seg in wing.segments:
                        name = seg.name
                        chord_grads = seg.chord_dist
                        twist_grads = seg.twist_angles
                        if chord_grads is not None:
                            for j, g in enumerate(chord_grads):
                                row[f"d{output}/d(chord_{name}_{j})"] = float(g)
                        if twist_grads is not None:
                            for j, g in enumerate(twist_grads):
                                row[f"d{output}/d(twist_{name}_{j})"] = float(g)

            grad_rows.append(row)

        gradients_df = pd.DataFrame(grad_rows)

    return forces_df, gradients_df


def save_results(
    plane: Airplane,
    state: State,
    forces_df: pd.DataFrame,
) -> None:
    DB = Database.get_instance()
    CASEDIR = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="LSPT",
    )
    filename = os.path.join(CASEDIR, "forces.lspt")
    forces_df.to_csv(filename, index=False, float_format="%.10f")

    plane.save()
    state.add_polar(
        polar=forces_df,
        polar_prefix="LSPT Potential",
        is_dimensional=True,
    )

    # Save the Forces
    # Add plane to database
    logging.info("Adding Results to Database")
    file_plane: str = os.path.join(DB.DB3D, plane.name, f"{plane.name}.json")
    _ = DB.load_vehicle(name=plane.name, file=file_plane)

    # Add Forces to Database
    DB.load_vehicle_solver_data(
        vehicle=plane,
        state=state,
        folder=plane.directory,
        solver="LSPT",
    )
