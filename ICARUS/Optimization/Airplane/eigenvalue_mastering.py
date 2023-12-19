# ICARUS IMPORTS
from ICARUS.Computation.Solvers.AVL.analyses.pertrubations import avl_dynamic_analysis_fd, process_avl_fd_res
from ICARUS.Core.types import FloatArray
from ICARUS.Environment.definition import EARTH_ISA
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Flight_Dynamics.trim import TrimNotPossible, TrimOutsidePolars
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Core.types import ComplexArray
from ICARUS.Core.types import FloatArray
from ICARUS.Environment.definition import Environment, EARTH_ISA

# 3D PARTY MODULES IMPORTS AND INITIALIZATIONS
from re import L
from time import time
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult

ii64 = np.iinfo(np.int64)
MAX_INT = ii64.max
f64 = np.finfo(np.float64)
MAX_FLOAT = float(f64.max)

# FLOW AND FLIGHT DYNAMICS PARAMETERS INITIALIZATION
desired_lateral_modes: ComplexArray = np.array([-30.0 + 0.0j, -1.0 + 4.0j, -1.0 - 4.0j, -0.5 + 0.0j], dtype=complex)
desired_longitudal_modes: ComplexArray = np.array([-3.0 - 4.0j, -3.0 + 4.0j, -0.7 - 0.4j, -0.7 + 0.4j], dtype=complex)
desired_longitudal_omegas = np.abs(desired_longitudal_modes)
desired_longitudal_zetas = -(desired_longitudal_modes.real) / desired_longitudal_omegas
desired_lateral_omegas = np.abs(desired_lateral_modes)
desired_lateral_zetas = -(desired_lateral_modes.real) / desired_lateral_omegas
UINF = 20
epsilons = {
    "u": 0.01,
    "w": 0.01,
    "q": 0.1,
    "theta": 0.01,
    "v": 0.01,
    "p": 0.1,
    "r": 0.1,
    "phi": 0.01,
}
solver2D = "Xfoil"

# USEFUL EXCEPTIONS


class Wrong_Axis_Mass_Position(Exception):
    pass


class Not_Supported_Design_Variable(Exception):
    pass


# FUNCTION THAT DYNAMICALLY CHANGES THE PLANE ACCORDING TO THE GIVEN DESIGN VARIABLES
def plane_set(
    design_vector: FloatArray,
    design_variables: list[dict],
    plane: Airplane,
):
    for j, dv in enumerate(design_variables):
        if dv["type"] == "mass_position":
            mass_ind = [i for i, m in enumerate(plane.masses) if m[2] == dv["mass_name"]]
            mass_ind = int(mass_ind[0])
            if dv["axis"] == "x":
                plane.masses[mass_ind][1][0] = design_vector[j]
            elif dv["axis"] == "y":
                plane.masses[mass_ind][1][1] = design_vector[j]
            elif dv["axis"] == "z":
                plane.masses[mass_ind][1][2] = design_vector[j]
            else:
                raise Wrong_Axis_Mass_Position
            plane.find_cg
            plane.find_inertia(plane.CG)
        elif dv["type"] == "tail_lever_arm":
            surf_ind = [i for i, s in enumerate(plane.surfaces) if s.name in ["elevator", "rudder"]]
            plane.surfaces[surf_ind[0]].x_origin = design_vector[j]
            plane.surfaces[surf_ind[1]].x_origin = design_vector[j]
        else:
            print(dv)
            raise Not_Supported_Design_Variable


# OBJECTIVE FUNCTION FOR THE PARAMETRIC STUDY OF AIRCRAFT DYNAMICX
def obj_fun(
    design_vector: FloatArray,
    design_variables: list[dict],
    plane: Airplane,
    env: Environment = EARTH_ISA,
    desired_longitudal_omegas=desired_longitudal_omegas,
    desired_longitudal_zetas=desired_longitudal_zetas,
    desired_lateral_omegas=desired_lateral_omegas,
    desired_lateral_zetas=desired_lateral_zetas,
    solver2D=solver2D,
):
    plane_set(design_vector, design_variables, plane)
    unstick = State(name="Unstick", airplane=plane, environment=env, u_freestream=UINF)
    unstick.add_all_pertrubations("Central", epsilons)

    try:
        avl_dynamic_analysis_fd(plane, unstick, solver2D)

    except (TrimNotPossible, TrimOutsidePolars):
        return np.inf
    df = process_avl_fd_res(plane, unstick)
    unstick.set_pertrubation_results(df)
    unstick.stability_fd()
    longitudal_eigs: ComplexArray = unstick.state_space.longitudal.eigenvalues
    lateral_eigs: ComplexArray = unstick.state_space.lateral.eigenvalues
    longitudal_omegas = unstick.state_space.longitudal.omegas
    longitudal_zetas = unstick.state_space.longitudal.zetas
    lateral_omegas = unstick.state_space.lateral.omegas
    lateral_zetas = unstick.state_space.lateral.zetas
    O = np.sum(np.abs(longitudal_omegas - desired_longitudal_omegas)) * np.sum(np.exp(-longitudal_zetas))
    return O


# DERIVATIVE CALCULATION
def jac_fun(
    design_vector: FloatArray, design_variables: list[dict], plane: Airplane, inc: FloatArray = np.array([1e-7])
):
    if len(inc) == 1:
        Jac = np.zeros(len(design_vector))
        for i, val in enumerate(design_vector):
            temp_dv = np.copy(design_vector)
            temp_dv[i] = val + inc
            O_f = obj_fun(temp_dv, design_variables, plane)
            temp_dv = np.copy(design_vector)
            temp_dv[i] = val - inc
            O_b = obj_fun(temp_dv, design_variables, plane)
            Jac[i] = (O_f - O_b) / (2 * inc)

        plane_set(design_vector, design_variables, plane)
    else:
        # Option for increment sensitivity analysis
        Jac = np.zeros((len(inc), len(design_vector)))
        for j, incr in enumerate(inc):
            for i, val in enumerate(design_vector):
                temp_dv = np.copy(design_vector)
                temp_dv[i] = val + incr
                # print("initial ", design_vector, "new ", temp_dv)
                O_f = obj_fun(temp_dv, design_variables, plane)
                temp_dv = np.copy(design_vector)
                temp_dv[i] = val - incr
                O_b = obj_fun(temp_dv, design_variables, plane)
                Jac[j, i] = (O_f - O_b) / (2 * incr)

                plane_set(design_vector, design_variables, plane)

    return Jac


# UNFINISHED CONSTRAIN FUNCTION
def cm_constr(design_vector, design_variables, plane):
    plane_set(design_vector, design_variables, plane)


# OPTIMIZER CLASS - USE OF SCIPY.OPTIMIZE.MINIMIZE
class Eigen_Optimizer:
    def __init__(
        self,
        plane_name: str,
        plane_fun: Callable,
        objective_fn: Callable[..., float],
        jacobian: Callable,
        x0: FloatArray,
        dvars: list[dict],
        bounds: list,
        maxtime_sec: float = MAX_FLOAT,
        max_iter: int = MAX_INT,
        max_function_call: int = MAX_INT,
        optimization_tolerance: float = 1e-6,
    ) -> None:
        # Basic Objects
        self.jacobian = jacobian
        self.objective_fn = objective_fn
        self.plane_name = plane_name
        self.x0 = x0
        self.plane_fun = plane_fun
        self.dvars = dvars
        self.bounds = bounds

        # Stop Parameters
        self.maxtime_sec: float = maxtime_sec
        self.max_function_call_count = max_function_call
        self.max_iter = max_iter
        self.tolerance = optimization_tolerance
        self._function_call_count: int = 0
        self._nit: int = 0
        self.current_plane = self.plane_fun(f"{self.plane_name}_0")

    def f(self, x: FloatArray, tab: bool = False) -> float:
        if self._function_call_count > self.max_function_call_count:
            raise StopIteration
        self._function_call_count += 1
        print(f"FUNCTION CALL {self._function_call_count}")
        self.current_plane = self.plane_fun(f"{self.plane_name}_{self._function_call_count}")

        if tab:
            print(f"\tCalculating OBJ {self._nit}")

        return self.objective_fn(x, self.dvars, self.current_plane)

    def j(self, x: FloatArray, tab: bool = True):
        if tab:
            print(f"\tCalculating J {self._nit}")
        return self.jacobian(x, self.dvars, self.current_plane)

    def callback(self, intermediate_result: OptimizeResult) -> None:
        # callback to terminate if maxtime_sec is exceeded
        self._nit += 1
        elapsed_time = time() - self.start_time

        if elapsed_time > self.maxtime_sec:
            print(elapsed_time)
            print(f"Fun: {intermediate_result.fun}")
            print(f"X : {intermediate_result.x}")
            print("TIME_EXCEEDED")
            raise StopIteration

        else:
            # you could print elapsed iterations and time
            print(f"Iteration Number: {self._nit}")

    def __call__(self) -> OptimizeResult:
        self.start_time = time()
        # set your initial guess to 'x0'
        # set your bounds to 'bounds'
        opt = minimize(
            self.f,
            x0=self.x0,
            # jac=self.j, neeeded for gradient-based optimization
            method="Nelder-Mead",
            callback=self.callback,
            tol=self.tolerance,
            options={"maxiter": self.max_iter, "disp": True},
            bounds=self.bounds,
        )
        return opt
