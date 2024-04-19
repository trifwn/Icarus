import jax
import numpy as np

from ICARUS.core.types import FloatArray
from ICARUS.mission.trajectory.trajectory import MissionTrajectory

# Use dynamical System Integrators
# from ICARUS.dynamical_systems.integrate import integrators


def RK4systems(
    t0: float,
    tend: float,
    dt: float,
    x0: jax.Array,
    v0: jax.Array,
    trajectory: MissionTrajectory,
    verbosity: int = 0,
) -> tuple[FloatArray, FloatArray, FloatArray, list[FloatArray]]:
    """Integrate the trajectory of an airplane using the RK4 method.

    Args:
        t0 (float): Start time
        tend (float): End time
        dt (float): Time step
        x0 (FloatArray): Initial position
        v0 (FloatArray): Initial velocity
        trajectory (Trajectory): Trajectory Object
        airplane (Mission_Vehicle): Airplane Object
        verbosity (int, optional): Verbosity. Defaults to 0.

    Returns:
        _type_: _description_
    """
    # print("Starting Simulation")
    # print(f"Initial X: {x0}")
    # print(f"Initial V: {v0}")
    x = [x0]
    v = [v0]
    t: FloatArray = np.arange(t0, tend + dt, dt)

    steps = round((tend - t0) / dt)
    success = True
    state = trajectory.get_control()
    states = [state]
    for i in np.arange(0, steps, 1):
        if verbosity:
            print(f"Step {i} of {steps} started")
            print(f"X Now is: {x[-1][0], x[-1][1]}")
            print(f"V Now is: {np.linalg.norm(v[-1])}")
            print(f"V_x Now is: {v[-1][0]} and V_y Now is: {v[-1][1]}")

        xi = np.array(x[i])
        vi = np.array(v[i])
        ti: float = t.tolist()[i]

        k1 = dt * trajectory.dxdt(xi, vi)
        l1, state1 = trajectory.dvdt(ti, xi, vi, state)
        l1 = dt * l1

        k2 = dt * trajectory.dxdt(xi + (k1 / 2), vi + l1 / 2)
        l2, state2 = trajectory.dvdt(ti, xi + (k1 / 2), vi + l1 / 2, state1)
        l2 = dt * l2

        k3 = dt * trajectory.dxdt(xi + (k2 / 2), vi + l2 / 2)
        l3, state3 = trajectory.dvdt(ti, xi + (k2 / 2), vi + l2 / 2, state2)
        l3 = dt * l3

        k4 = dt * trajectory.dxdt(xi + k3, vi + l3)
        l4, state4 = trajectory.dvdt(ti, xi + k3, vi + l3, state3)
        l4 = dt * l4

        k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        l = (l1 + 2 * l2 + 2 * l3 + l4) / 6

        # Average the states
        state = (state1 + 2 * state2 + 2 * state3 + state4) / 6
        states.append(state)

        x.append(xi + k)
        v.append(vi + l)

        if np.isnan(xi).any() or bool(np.isnan(vi).any()) | bool(np.isinf(xi).any()) or np.isinf(vi).any():
            print(f"Blew UP at step: {i}\tTime {t[i]}")
            print(f"\tMax Distance: {x[-1][0]}")
            success = False
            break
        if x[-1][1] < trajectory.operating_floor:
            print(f"Airplane Crash at step: {i}\tTime {t[i]}")
            print(f"\tMax Distance: {x[-1][0]} ")
            success = False
            break
        if x[-1][0] < 0:
            print(f"Airplane Went to negative at step: {i}\tTime {t[i]}")
            print(f"\tMax Distance: {x[-1][0]}")
            success = False
            break
        if v[-1][0] < 0:
            print(f"Negative velocity at step: {i} \t {v[-1][0]}")
            print(f"Airplane Goes Backwords at step:{i}\tTime {t[i]}")
            print(f"\tMax Distance: {x[-1][0]}")
            success = False
            break
    if success:
        print(f"Simulation Completed Successfully at time {t[-1]}\tTime {t[i]}Max Distance: {x[-1][0]}")
    else:
        # Return the last valid state
        pass
        # print(x[-1])
        # x[-1] = x[-2]/2
        # print(f"Simulation Failed                        Max Distance: {x[-1][0]}")
    #   for statei, xi, vi, ti in zip(states, x, v, t):
    #       trajectory.record_state(ti, xi, vi, *statei)
    # Plot Trajectory
    # trajectory.clear_history()
    # for statei, xi, vi, ti in zip(states, x, v, t):
    #     trajectory.record_state(ti, xi, vi, *statei)
    return t, np.array(x), np.array(v), states


from scipy.integrate import RK45


def RK45_scipy_integrator(
    t0: float,
    tend: float,
    dt: float,
    x0: jax.Array,
    v0: jax.Array,
    trajectory: MissionTrajectory,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    status = "Successfull"

    x = [np.array([*x0, *v0])]
    t = [t0]

    r = RK45(fun=trajectory.timestep, t0=t0, y0=[*x0, *v0], t_bound=tend, max_step=dt, vectorized=False)
    while r.status == "running":
        r.step()
        t.append(r.t)
        x.append(r.y)
        if np.isnan(x[-1]).any():
            break
    xs = np.array(x)
    positions = xs[:, :2]
    velocities = xs[:, 2:]

    print(f"Simulation Completed Successfully at time {t[-1]}       Max Distance: {x[-1][0]}")
    print(status)
    return np.array(t), positions, velocities


from scipy.integrate import solve_ivp


def scipy_ivp_integrator(
    t0: float,
    tend: float,
    x0: FloatArray,
    v0: FloatArray,
    trajectory: MissionTrajectory,
) -> tuple[FloatArray, FloatArray, FloatArray]:

    t = [t0]
    r = solve_ivp(trajectory.timestep, (t0, tend), [*x0, *v0], t_eval=t, vectorized=False)
    x = r.y

    print(f"Simulation Completed Successfully at time {t[-1]}       Max Distance: {x[-1][0]}")
    return np.array(t), x[:, :2], x[:, 2:]
