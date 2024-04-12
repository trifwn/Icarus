import jax
from jax.debug import print as jprint
import jax.numpy as jnp

from ICARUS.mission.trajectory.integrate import RK4systems

# Get the percision of the jax library
print()
print('Jax has been configured to use the following devices: ', jax.devices())
jax.config.update("jax_enable_x64", True)
# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')
print('Jax has been configured to use the following devices: ', jax.devices())
jax.config.update('jax_platform_name', 'cpu')
print(jax.numpy.ones(3).devices()) # TFRT_CPU_0

from ICARUS import APPHOME
from ICARUS.propulsion.engine import Engine
from ICARUS.database import DB
from ICARUS.vehicle.plane import Airplane
from ICARUS.mission.trajectory.trajectory import MissionTrajectory
from ICARUS.geometry.cubic_splines import CubicSpline_factory
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController, DiscreteTerminatingEvent, DirectAdjoint
from ICARUS.mission.mission_vehicle import MissionVehicle


engine_dir = f"{APPHOME}/Data/Engine/Motor_1/"
engine = Engine()
engine.load_data_from_df(engine_dir)


plane: Airplane = DB.get_vehicle('final_design')

mission_plane = MissionVehicle(
    plane,
    engine,
    solver= "AVL"
)

operating_floor = 12.5
t0 = 0.
x0 =  jnp.array([0., 20.])

# Polynomial Trajectory
coeffs = jnp.array(
    [
        x0[1],
        0.2,
        -1/300
    ]
)

args = None
solver = Tsit5(scan_kind="bounded")
ts = jnp.linspace(0, 500, 101)
saveat = SaveAt(ts=ts)
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5, dtmin= 1e-4)

def default_terminating_event_fxn(state, **kwargs):
    terms = kwargs.get("terms", lambda a, x, b: x)
    return jnp.any(jnp.isnan(terms.vf(state.tnext, state.y, 0)))
terminating_event =  DiscreteTerminatingEvent(default_terminating_event_fxn)

t1 = ts[-1]
v0_mag = 20.

@jax.jit
def fun(y, tend):

    x0 =  jnp.array([0., 20.])
    x =  jnp.linspace(0, 15000, y.shape[0] + 2)
    y = jnp.hstack([x0[1], y, operating_floor])
    spline_i = CubicSpline_factory(x,y)

    traj_spl = MissionTrajectory(
        "CubicSpline", 
        spline_i, 
        vehicle=mission_plane,
        verbosity= 2,
        operating_floor= operating_floor
    )

    g1 = jnp.arctan(
        traj_spl.dy_dx(x0[0])
    )

    v0 = jnp.array([jnp.cos(g1), jnp.sin(g1)]) * v0_mag
    y0 = jnp.hstack([x0, v0])
    term = ODETerm(traj_spl.timestep)

    solution = diffeqsolve(term, solver, t0=t0, t1=tend, dt0=0.1, y0=y0,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                discrete_terminating_event= terminating_event,
                max_steps= 10000,
        )
    ts = jnp.array(solution.ts)
    ys = jnp.array(solution.ys)
    jprint("Control Point X {}", x)
    jprint("Control Point Y {}", y)
    x = jnp.nan_to_num(ys[:,0], nan = 0 , posinf= 0, neginf= 0)
    y = jnp.nan_to_num(ys[:,1], nan = 0 , posinf= 0, neginf= 0)
    u = jnp.nan_to_num(ys[:,2], nan = 0 , posinf= 0, neginf= 0)
    v = jnp.nan_to_num(ys[:,3], nan = 0 , posinf= 0, neginf= 0)
    # jprint("X: {}", x)
    # jprint("Y: {}", y)
    # jprint("t: {}", ts)
    jprint("\033[92mReturning {} \033[0m", -jnp.max(x))
    return -jnp.max(x)/1000

print("\n\n\n\n\n\n")
print("Compiling the function. First Call will be slow.")
import time
time_s = time.time() 

solution = fun(y = jnp.array([20.1, 20.1, 20.1, 20.1,20.1, 20.1, 20.1, 20.1,20.1, 20.1, 20.1, 20.1]), tend = ts[-1])

# Calculate the jacrev for the arg y

print(f"Time taken: {time.time() - time_s}")
print("\n\n\n\n\n\n")
# Time to calculate the jacobian
print("Compiling and Calculating the Jacobian. First Call will be slow.")
time_s = time.time()

@jax.jit
def jacobian(x, tend):
    J =  jax.jacrev(fun, argnums=0)(x, tend)
    jprint("Jacobian: {}", J)
    return J

JAC= jacobian(x = jnp.array([20.1, 20.1, 20.1, 20.1,20.1, 20.1, 20.1, 20.1,20.1, 20.1, 20.1, 20.1]), tend = ts[-1])
print(f"Time taken: {time.time() - time_s}")
print("\n\n\n\n\n\n")

@jax.jit
def hessian(x, tend):
    return jax.hessian(fun, argnums=0)(x, tend)

print("Compiling and Calculating the Hessian. First Call will be slow.")
time_s = time.time()

H = hessian(x = jnp.array([20.1, 20.1, 20.1, 20.1,20.1, 20.1, 20.1, 20.1,20.1, 20.1, 20.1, 20.1]), tend = ts[-1])
print(f"Time taken: {time.time() - time_s}")
print("\n\n\n\n\n\n")


from scipy.optimize import minimize

print("Optimizing")
res_splines = minimize(
    fun,
    x0 = jnp.array([20.1, 20.1, 20.1, 20.1,20.1, 20.1, 20.1, 20.1,20.1, 20.1, 20.1, 20.1]),
    jac = jacobian,
    # hess = hessian,
    method='CG',
    # method = "Powell",
    options={ 'maxiter' : 30000, 'disp': True},
    args = (ts[-1],)
)

y  = res_splines.x
x = jnp.linspace(0, 15000, len(y)+2)
y = jnp.hstack([x0[1], y, operating_floor])

import jaxopt

# solverLBFGS = jaxopt.LBFGS(fun=fun, maxiter=30000)
# solverGD = jaxopt.GradientDescent(fun=fun, maxiter=5000, stepsize=-5)
# solverNCG = jaxopt.NonlinearCG(fun=fun, method="polak-ribiere", maxiter=5000)


# res = solverNCG.run(res_splines.x, tend = ts[-1])
# print(res)

spline_best = CubicSpline_factory(x,y)
trajectory_best =  MissionTrajectory(
        "Optimized CubicSpline", 
        spline_best, 
        vehicle=mission_plane,
        verbosity= 2,
        operating_floor= operating_floor
    )
gamma = jnp.arctan( trajectory_best.dy_dx(x0[0]) )
v0 = jnp.array([jnp.cos(gamma), jnp.sin(gamma)]) * v0_mag

t, xs, vs, states = RK4systems(t0, ts[-1], 0.1, x0, v0, trajectory_best)

trajectory_best.clear_history()
for statei, xi, vi, ti in zip(states, xs, vs, t):
    trajectory_best.record_state(ti, xi, vi, *statei)
# Plot Trajectory
trajectory_best.plot_history()
import matplotlib.pyplot as plt
plt.show(block = True)
