from time import time

import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from scipy.integrate import solve_ivp

from ICARUS.dynamical_systems import DynamicalSystem
from ICARUS.dynamical_systems import SecondOrderSystem
from ICARUS.dynamical_systems.integrate import BackwardEulerIntegrator
from ICARUS.dynamical_systems.integrate import ForwardEulerIntegrator
from ICARUS.dynamical_systems.integrate import GaussLegendreIntegrator
from ICARUS.dynamical_systems.integrate import NewmarkIntegrator
from ICARUS.dynamical_systems.integrate import RK4Integrator
from ICARUS.dynamical_systems.integrate import RK45Integrator


def plot_results(x_data: dict[str, Array], t_data: dict[str, Array]) -> None:
    # Plot the results
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_title("Simulation Results")
    for key in list(x_data.keys()):
        # Compute cap so that we plot 1000 points at most
        cap = max(1, int(len(t_data[key]) / 1000))
        ax.plot(t_data[key][::cap], x_data[key][:, 0][::cap] * 180 / np.pi, label=key)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.legend()


def test_all_integrators(
    system: DynamicalSystem,
    x0: Array,
    t0: float,
    t_end: float,
    dt0: float,
    compare_with_scipy: bool = False,
) -> tuple[dict[str, Array], dict[str, Array]]:
    if isinstance(system, SecondOrderSystem):
        system2: DynamicalSystem = system.convert_to_first_order()
    else:
        system2 = system

    integrator_rk4 = RK4Integrator(dt0, system2)
    integrator_feuler = ForwardEulerIntegrator(dt0, system2)
    integrator_beuler = BackwardEulerIntegrator(dt0, system2)
    integrator_rk45 = RK45Integrator(dt0, system2)
    integrator_gauss_legendre = GaussLegendreIntegrator(dt0, system2, n=4)
    # integrator_crank_nicolson = CrankNicolsonIntegrator(dt0, system2)

    integrators = [
        integrator_feuler,
        integrator_rk4,
        integrator_rk45,
        integrator_beuler,
        integrator_gauss_legendre,
        # integrator_crank_nicolson,
    ]

    if isinstance(system, SecondOrderSystem):
        newmark = NewmarkIntegrator(dt0, system, gamma=0.5, beta=0.25)
        integrators.append(newmark)

    # Simulate the system using all the integrators
    x_data = {}
    t_data = {}

    for integrator in integrators:
        print(f"Simulating using {integrator.name} integrator")
        time_s = time()
        t_data[integrator.name], x_data[integrator.name] = integrator.simulate(
            x0,
            t0,
            t_end,
        )
        time_e = time()
        print(f"\tSimulated using {integrator.name} integrator")
        print(f"\tTime taken: {time_e - time_s} seconds")

    if compare_with_scipy:
        print("Simulating using scipy RK45")
        time_s = time()
        sol = solve_ivp(
            system,
            [t0, t_end],
            x0,
            method="RK45",
            t_eval=np.linspace(0, t_end, 1000),
            rtol=1e-6,
            atol=1e-6,
        )
        t_data["scipy_rk45"] = sol.t
        x_data["scipy_rk45"] = sol.y.T
        time_e = time()
        print("\tSimulated using scipy RK45")
        print(f"\tTime taken: {time_e - time_s} seconds")

    plot_results(x_data, t_data)
    return x_data, t_data
