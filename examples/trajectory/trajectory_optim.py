import os
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import optimistix as optx
from diffrax import DirectAdjoint  # Tsit5,; BacksolveAdjoint,
from diffrax import Dopri8
from diffrax import ODETerm
from diffrax import PIDController
from diffrax import SaveAt
from diffrax import diffeqsolve
from diffrax._event import DiscreteTerminatingEvent
from jax.debug import print as jprint
from jaxtyping import Array
from jaxtyping import Float

from ICARUS import INSTALL_DIR
from ICARUS.database import Database
from ICARUS.geometry.cubic_splines import CubicSpline_factory
from ICARUS.mission.mission_vehicle import MissionVehicle
from ICARUS.mission.trajectory.integrate import RK4systems
from ICARUS.mission.trajectory.trajectory import MissionTrajectory
from ICARUS.propulsion.engine import Engine
from ICARUS.vehicle.plane import Airplane

# Get the percision of the jax library
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
print("Jax has been configured to use the following devices: ", jax.devices())
# Global flag to set a specific platform, must be used at startup.
print(jax.numpy.ones(3).devices())


DB = Database(os.path.join(INSTALL_DIR, "Data"))

# #  Load Plane and Engine
engine_dir = os.path.join(DB.HOMEDIR, "Engine", "Motor_1")
engine = Engine()
engine.load_data_from_df(engine_dir)
# engine.plot_engine_map()


plane: Airplane = DB.get_vehicle("final")
# plane.visualize(annotate=True)
mission_plane = MissionVehicle(plane, engine, solver="AVL")

operating_floor = 12.5
T0 = 0.0
X0 = jnp.array([0.0, 60.0])
V0_MAG = 25.0
TEND = 90.0
TRAJECTORY_MAX_DIST = 3000.0
NUM_SPLINE_CONTROL_POINTS = 45


solver = Dopri8(scan_kind="bounded")
ts = jnp.linspace(T0, TEND, 1001)
saveat = SaveAt(ts=ts)
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5, dtmin=1e-4)


def default_terminating_event_fxn(state: Any, **kwargs: Any) -> jnp.ndarray:
    terms = kwargs.get("terms", lambda a, x, b: x)
    return jnp.any(jnp.isnan(terms.vf(state.tnext, state.y, 0)))


terminating_event = DiscreteTerminatingEvent(default_terminating_event_fxn)

dim = (NUM_SPLINE_CONTROL_POINTS,)


def fun(y: Float[Array, "{dim}"], *args: Any) -> Float[Array, "1"]:
    x0 = X0
    x = jnp.linspace(0, TRAJECTORY_MAX_DIST, y.shape[0] + 1)
    y = jnp.hstack([x0[1], y])
    spline_i = CubicSpline_factory(x, y)

    traj_spl = MissionTrajectory(
        "CubicSpline",
        spline_i,
        vehicle=mission_plane,
        verbosity=2,
        operating_floor=operating_floor,
    )

    g1 = jnp.arctan(traj_spl.dy_dx(x0[0]))

    v0 = jnp.array([jnp.cos(g1), jnp.sin(g1)]) * V0_MAG
    y0 = jnp.hstack([x0, v0])
    term = ODETerm(traj_spl.timestep)

    solution = diffeqsolve(
        term,
        solver,
        t0=T0,
        t1=TEND,
        dt0=0.1,
        y0=y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        discrete_terminating_event=terminating_event,
        max_steps=10000,
        adjoint=DirectAdjoint(),
    )
    # ts = jnp.array(solution.ts)
    ys = jnp.array(solution.ys)
    jprint("\tControl Point X {}", x)
    jprint("\tControl Point Y {}", y)
    x = jnp.nan_to_num(ys[:, 0], nan=0, posinf=0, neginf=0)
    y = jnp.nan_to_num(ys[:, 1], nan=0, posinf=0, neginf=0)
    # u = jnp.nan_to_num(ys[:, 2], nan=0, posinf=0, neginf=0)
    # v = jnp.nan_to_num(ys[:, 3], nan=0, posinf=0, neginf=0)
    # jprint("X: {}", x)
    # jprint("Y: {}", y)
    # jprint("t: {}", ts)
    jprint("\033[92mReturning {} \033[0m", -jnp.max(x))
    return -x[-1] / TRAJECTORY_MAX_DIST


fun = jax.jit(fun)


def jacobian(y: Float[Array, "dim"], *args: Any) -> Float[Array, "dim"]:
    J = jax.jacrev(fun, argnums=0)(y)
    jprint("\033[91mJacobian: {} \033[0m", J)
    return J


jacobian = jax.jit(jacobian)


def compute_and_plot(y: Float[Array, "..."]) -> None:
    x0 = X0
    x = jnp.linspace(0, TRAJECTORY_MAX_DIST, y.shape[0] + 1)
    y = jnp.hstack([x0[1], y])

    some_spl = CubicSpline_factory(x, y)
    trajectory_best = MissionTrajectory(
        "Optimized CubicSpline",
        some_spl,
        vehicle=mission_plane,
        verbosity=2,
        operating_floor=operating_floor,
    )
    gamma = jnp.arctan(trajectory_best.dy_dx(X0[0]))
    v0 = jnp.array([jnp.cos(gamma), jnp.sin(gamma)]) * V0_MAG

    t, xs, vs, states = RK4systems(T0, TEND, 0.1, X0, v0, trajectory_best)

    trajectory_best.clear_history()
    for statei, xi, vi, ti in zip(states, xs, vs, t):
        trajectory_best.record_state(ti, xi, vi, *statei)
    # Plot Trajectory
    trajectory_best.plot_history()
    plt.show(block=True)


# # Function Call


# print("Compiling the function. First Call will be slow.")
# import time
# time_s = time.time()

# y_test = jnp.repeat(60., NUM_SPLINE_CONTROL_POINTS)
# solution = fun(y = y_test)

# # Calculate the jacrev for the arg y

# print(f"Time taken: {time.time() - time_s}")


# print("Compiling and Calculating the Jacobian. First Call will be slow.")
# time_s = time.time()


# y_test = jnp.repeat(15., NUM_SPLINE_CONTROL_POINTS)
# JAC= jacobian(y = y_test)
# print(f"Time taken: {time.time() - time_s}")


# # Optimization JAX
# solver_optimistix = optx.BestSoFarMinimiser(
#     optx.GradientDescent(
#         rtol=1e-12,
#         atol=1e-12,
#         learning_rate=1e-1,
#         # search = optx.BacktrackingArmijo(decrease_factor=0.9, slope=0.1, step_init=1.)
#     ),
# )


solver_optax = optx.BestSoFarMinimiser(
    optx.OptaxMinimiser(optax.adam(1e-3, eps=1e-10), rtol=1e-12, atol=1e-12),
)

# y_initial = jnp.array([
#     58.52307106 ,
#     58.58927354 ,
#     58.56428532 ,
#     57.98786009 ,
#     54.5685268,
#     47.50944514 ,
#     28.9483089  ,
#     23.87737692 ,
#     94.65244459 ,
#     25.41280581,
# ])
y_initial = jnp.repeat(60.0, NUM_SPLINE_CONTROL_POINTS)

print("Optimizing with Optimistix.")
res_splines = optx.minimise(
    fn=fun,
    y0=y_initial,
    solver=solver_optax,
    max_steps=300000,
    throw=False,
)

print(res_splines.stats)
y = res_splines.value
compute_and_plot(y)


# # Optimization Scipy


# from scipy.optimize import minimize

# res_splines = minimize(
#     fun =fun,
#     jac = jacobian,
#     x0 = np.array([20, 20 , 20, 20, 20, 20,20, 20, 20,20, 20, 20,]),
#     method='COBYLA',
#     options={'disp': True, 'maxiter' : 3000},
# )


# print(res_splines.x)
# y = res_splines.x
# compute_and_plot(y)
