from __future__ import annotations

from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
from pandas import DataFrame

from ICARUS.aero import AerodynamicLoads
from ICARUS.aero.aerodynamic_state import AerodynamicState

if TYPE_CHECKING:
    from jax import Array
    from ICARUS.aero import LSPT_Plane
    from ICARUS.flight_dynamics import State
    from ICARUS.core.types import FloatArray

from . import get_LHS
from . import get_RHS


def run_vlm_analysis(plane: LSPT_Plane, state: State, angles: list[float] | Array | FloatArray) -> DataFrame:
    """Run complete VLM analysis workflow integrating all components.

    This method implements the complete workflow:
    1. Factorize VLM matrices
    2. Calculate gammas and w_induced for each angle
    3. Distribute calculations to strips
    4. Calculate potential loads
    5. Calculate viscous loads

    Args:
        plane: LSPT_Plane object containing geometry
        state: Flight state containing environment data
        angles: List or array of angles of attack in degrees
            Returns:
        DataFrame containing analysis results
    """

    # Step 1: Factorize VLM system matrices
    A, A_star = get_LHS(plane)
    A_LU, A_piv = jax.scipy.linalg.lu_factor(A)

    # Initialize result arrays
    results = {
        "AoA": [],
        "Lift_Potential": [],
        "Drag_Potential": [],
        "Lift_Viscous": [],
        "Drag_Viscous": [],
        "CL": [],
        "CD": [],
        "CL_2D": [],
        "CD_2D": [],
    }

    # load_data: list[AerodynamicLoads] = []

    aerodynamic_state = AerodynamicState(
        airspeed=state.u_freestream,
        altitude=state.environment.altitude,
        density=state.environment.air_density,
        mach=0.0,  # Assuming incompressible flow
        # Positional State
        alpha=0.0,
        beta=0.0,
        rate_P=0.0,
        rate_Q=0.0,
        rate_R=0.0,
    )

    for angle in angles:
        # Step 0: Update aerodynamic state
        aerodynamic_state.alpha = angle
        Q = aerodynamic_state.velocity_vector_jax
        dynamic_pressure = aerodynamic_state.dynamic_pressure

        # Step 1: Update plane with current aerodynamic state
        # This basically sets the angle of the wake and the angle of attack
        # plane.update_aerodynamic_state(aerodynamic_state)

        # Step 2: Calculate RHS
        RHS = get_RHS(plane, Q)

        if jnp.any(jnp.isnan(RHS)):
            raise ValueError("NaN values found in RHS. Check aerodynamic state or plane geometry.")

        if jnp.any(jnp.isnan(A_star)):
            raise ValueError("NaN values found in A_star. Check factorization of VLM matrices.")

        if jnp.any(jnp.isnan(A_LU)):
            raise ValueError("NaN values found in A_LU. Check factorization of VLM matrices.")

        # Step 3: Solve for circulations using factorized system
        gammas = jax.scipy.linalg.lu_solve((A_LU, A_piv), RHS)
        w_induced = jnp.matmul(A_star, gammas)

        if jnp.any(jnp.isnan(gammas)):
            raise ValueError("NaN values found in gammas. Check factorization or RHS calculation.")

        # Step 4: Create AerodynamicLoads
        loads = AerodynamicLoads(plane=plane)

        # Step 5: Distribute gamma calculations to strips
        loads.distribute_gamma_calculations(gammas, w_induced)
        print(f"Angle of Attack: {angle} degrees")

        # Step 6: Calculate potential loads
        total_lift_potential, total_drag_potential, total_moment_potential = loads.calculate_potential_loads(
            aerodynamic_state,
        )
        print(f"\tTotal Lift (Potential): {total_lift_potential:.2f} N")
        print(f"\tTotal Drag (Potential): {total_drag_potential:.2f} N")
        print(f"\tTotal Moment (Potential): {total_moment_potential:.2f} Nm")
        # Step 7: Calculate viscous loads
        total_lift_viscous, total_drag_viscous, total_moment_viscous = loads.calculate_viscous_loads(
            aerodynamic_state,
        )

        # Calculate coefficients
        CL = total_lift_potential / (dynamic_pressure * plane.S)
        CD = total_drag_potential / (dynamic_pressure * plane.S)

        CL_2D = total_lift_viscous / (dynamic_pressure * plane.S)
        CD_2D = total_drag_viscous / (dynamic_pressure * plane.S)

        # Store results
        results["AoA"].append(float(angle))
        results["Lift_Potential"].append(float(total_lift_potential))
        results["Drag_Potential"].append(float(total_drag_potential))
        results["Lift_Viscous"].append(float(total_lift_viscous))
        results["Drag_Viscous"].append(float(total_drag_viscous))
        results["CL"].append(float(CL))
        results["CD"].append(float(CD))
        results["CL_2D"].append(float(CL_2D))
        results["CD_2D"].append(float(CD_2D))

    return DataFrame(results)
