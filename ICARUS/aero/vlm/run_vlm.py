from __future__ import annotations

from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
import pandas as pd

from ICARUS.aero import AerodynamicLoads
from ICARUS.aero.aerodynamic_state import AerodynamicState

if TYPE_CHECKING:
    from jax import Array
    from ICARUS.aero import LSPT_Plane
    from ICARUS.flight_dynamics import State
    from ICARUS.core.types import FloatArray

from . import get_LHS
from . import get_RHS


def run_vlm_analysis(plane: LSPT_Plane, state: State, angles: list[float] | Array | FloatArray) -> pd.DataFrame:
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

    results = pd.DataFrame()
    for angle in angles:
        # Step 0: Update aerodynamic state
        aerodynamic_state.alpha = angle
        Q = aerodynamic_state.velocity_vector_jax

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

        print(f"\tTotal Lift (Viscous): {total_lift_viscous:.2f} N")
        print(f"\tTotal Drag (Viscous): {total_drag_viscous:.2f} N")
        print(f"\tTotal Moment (Viscous): {total_moment_viscous:.2f} Nm")

        # Store results
        aoa_data = loads.to_dataframe(aerodynamic_state=aerodynamic_state, plane=plane)

        # Add series to results DataFrame
        results = pd.concat([results, aoa_data.to_frame().T])

    return results.sort_values(by = "AoA")
