from __future__ import annotations

from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp

from ICARUS.aero import AerodynamicLoads
from ICARUS.aero import AerodynamicResults
from ICARUS.aero import AerodynamicState

if TYPE_CHECKING:
    from jax import Array
    from ICARUS.flight_dynamics import State
    from ICARUS.core.types import FloatArray

from . import get_LHS
from . import get_RHS

from ICARUS.aero import LSPT_Plane
from ICARUS.vehicle.airplane import Airplane


def run_vlm_polar_analysis(
    plane: LSPT_Plane | Airplane, state: State, angles: list[float] | Array | FloatArray
) -> AerodynamicResults:
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

    if isinstance(plane, Airplane):
        lspt_plane = LSPT_Plane(plane=plane)
    elif isinstance(plane, LSPT_Plane):
        lspt_plane = plane
    else:
        raise TypeError("plane must be an instance of LSPT_Plane or Airplane")

    # Step 1: Factorize VLM system matrices
    A, A_star = get_LHS(lspt_plane)
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

    aero_results = AerodynamicResults(plane=lspt_plane)
    for angle in angles:
        aerodynamic_state = aerodynamic_state.copy()
        # Step 0: Update aerodynamic state
        aerodynamic_state.alpha = angle
        Q = aerodynamic_state.velocity_vector_jax

        # Step 1: Update plane with current aerodynamic state
        # This basically sets the angle of the wake and the angle of attack
        # plane.update_aerodynamic_state(aerodynamic_state)

        # Step 2: Calculate RHS
        RHS = get_RHS(lspt_plane, Q)

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
        loads = AerodynamicLoads(plane=lspt_plane)

        # Step 5: Distribute gamma calculations to strips
        loads.distribute_gamma_calculations(gammas, w_induced)

        # Step 6: Calculate potential loads
        _ = loads.calculate_potential_loads(
            aerodynamic_state,
        )
        # Step 7: Calculate viscous loads
        _ = loads.calculate_viscous_loads(
            aerodynamic_state,
        )

        aero_results.add_result(state=aerodynamic_state, loads=loads)
    return aero_results
