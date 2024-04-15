from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.debug import print as jprint
from jaxopt import Broyden
from jaxtyping import Array
from jaxtyping import Float
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray

from ICARUS.core.types import FloatArray
from ICARUS.mission.mission_vehicle import MissionVehicle


class MissionTrajectory:
    def __init__(
        self,
        name: str,
        trajectory_function: Callable[[jax.Array], jax.Array],
        vehicle: MissionVehicle,
        operating_floor: float,
        verbosity: int = 2,
    ) -> None:
        self.name: str = name

        # History of the trajectory
        self.times: list[float] = []

        self.positions: dict[str, list[float]] = {}
        self.positions["x"] = []
        self.positions["y"] = []

        self.velocities: dict[str, list[float]] = {}
        self.velocities["x"] = []
        self.velocities["y"] = []

        self.accelerations: dict[str, list[float]] = {}
        self.accelerations["x"] = []
        self.accelerations["y"] = []

        self.forces: dict[str, list[float]] = {}
        self.forces["x"] = []
        self.forces["y"] = []
        self.forces["lift"] = []
        self.forces["thrust"] = []
        self.forces["drag"] = []
        self.forces["torque"] = []

        self.control_inputs: dict[str, list[float]] = {}
        self.control_inputs["engine_amps"] = []
        self.control_inputs["aoa"] = []

        self.orientation: dict[str, list[float]] = {}
        self.orientation["alpha"] = []
        self.orientation["aoa"] = []
        self.orientation["gamma"] = []

        self.vehicle = vehicle

        self.operating_floor = operating_floor
        self.verbosity = verbosity

        self.fun: Callable[[jax.Array], jax.Array] = trajectory_function
        self.y = self.fun

        # Jax Functions
        self.y_jit = jax.jit(self.y)
        self.dy_dx = jax.jit(jax.grad(self.fun))
        self.d2y_dx2 = jax.jit(jax.grad(self.dy_dx))
        self.broyden = Broyden(
            self.f_to_trim,
            has_aux=False,
            verbose=False,
            stop_if_linesearch_fails=False,
            tol=1e-7,
            maxiter=1000,
            history_size=10,
        )
        self.broyden.verbose = 0

    def clear_history(self) -> None:
        self.times = []

        self.positions = {}
        self.positions["x"] = []
        self.positions["y"] = []

        self.velocities = {}
        self.velocities["x"] = []
        self.velocities["y"] = []

        self.accelerations = {}
        self.accelerations["x"] = []
        self.accelerations["y"] = []

        self.forces = {}
        self.forces["x"] = []
        self.forces["y"] = []
        self.forces["lift"] = []
        self.forces["thrust"] = []
        self.forces["drag"] = []
        self.forces["torque"] = []

        self.control_inputs = {}
        self.control_inputs["engine_amps"] = []
        self.control_inputs["aoa"] = []

        self.orientation = {}
        self.orientation["alpha"] = []
        self.orientation["aoa"] = []
        self.orientation["gamma"] = []

    def record_state(
        self,
        t: float,
        X: Float[Array, "2"],
        V: Float[Array, "2"],
        aoa: float,
        amps: float,
    ) -> None:
        dh_dx = self.dy_dx(X[0])
        dh2_dx2 = self.d2y_dx2(X[0])
        m = self.vehicle.mass
        G = 9.81

        alpha = jnp.deg2rad(aoa) + jnp.arctan(dh_dx)

        x = X[0]
        y = X[1]
        u = V[0]
        w = V[1]
        alpha = jnp.rad2deg(aoa)
        gamma = jnp.rad2deg(dh_dx)

        thrust = self.vehicle.motor.thrust(velocity=jnp.linalg.norm(V), current=amps)
        lift, drag, torque = self.vehicle.get_aerodynamic_forces(jnp.linalg.norm(V), aoa)
        # Compute the forces
        thrust_x = thrust * jnp.cos(alpha)
        thrust_y = thrust * jnp.sin(alpha)

        acc_x = (
            thrust_x  # Thrust
            - lift * dh_dx / jnp.sqrt(dh_dx**2 + 1)  # Lift
            - drag / jnp.sqrt(dh_dx**2 + 1)  # Drag
            # +  Elevator drag due to control
        ) / m

        acc_y = (
            thrust_y  # Thrust
            + lift / jnp.sqrt(dh_dx**2 + 1)  # Lift
            - drag * dh_dx / jnp.sqrt(dh_dx**2 + 1)  # Drag
            - m * G  # Weight
            # + Elevator lift due to control
        ) / m

        ACC = jnp.array([acc_x, acc_y], dtype=float).squeeze()

        fx = ACC[0] * m
        fy = ACC[1] * m

        L = lift
        T = thrust
        D = drag
        TORQUE = torque

        step_successful = jnp.isnan(ACC).any()
        if not step_successful:
            # Register the state
            self.times.append(t)
            self.positions["x"].append(x)
            self.positions["y"].append(y)

            self.velocities["x"].append(u)
            self.velocities["y"].append(w)

            self.accelerations["x"].append(acc_x)
            self.accelerations["y"].append(acc_y)

            self.forces["x"].append(fx)
            self.forces["y"].append(fy)

            self.forces["lift"].append(L)
            self.forces["thrust"].append(T)
            self.forces["drag"].append(D)
            self.forces["torque"].append(TORQUE)

            self.control_inputs["aoa"].append(aoa)
            self.control_inputs["engine_amps"].append(amps)

            self.orientation["alpha"].append(alpha)
            self.orientation["aoa"].append(aoa)
            self.orientation["gamma"].append(gamma)
        else:
            jprint("State not appended")
            jprint("F: {}", ACC)
            jprint("{}", jnp.isnan(ACC).any())

    @partial(jax.jit, static_argnums=(0,))
    def get_control(
        self, 
    ) -> Float[Array, "2"]:
        
        aoa = jnp.array([0])
        amps = jnp.array([30])
        return jnp.array([aoa, amps])

    @partial(jax.jit, static_argnums=(0,))
    def dxdt(self, X: Float[Array, "2"], V: Float[Array, "2"]) -> Float[Array, "2"]:
        return V

    # Get the aerodynamic forces
    @partial(jax.jit, static_argnums=(0,))
    def f_to_trim(
        self,
        aoa: float | Float[Array, "1"],
        thrust: float | Float[Array, "1"],
        V: Float[Array, "2"],
        dh_dx: float,
        dh2_dx2: float,
    ) -> float:
        """

        Args:
            aoa (float): In degrees

        """
        G = 9.81
        m = self.vehicle.mass

        lift, _, _ = self.vehicle.get_aerodynamic_forces(jnp.linalg.norm(V), aoa)
        alpha = jnp.deg2rad(aoa) + jnp.arctan(dh_dx)
        residual = (
            thrust * (jnp.sin(alpha) - dh_dx * jnp.cos(alpha))
            + lift * (dh_dx**2 + 1) / jnp.sqrt(dh_dx**2 + 1)
            - m * G
            - m * dh2_dx2 * V[0] ** 2
        )
        # jprint("\tResidual: {}, AoA: {}, alpha: {}, gamma {}, dhdx  {}",residual,aoa,jnp.rad2deg(alpha), jnp.rad2deg(jnp.arctan(dh_dx)), dh_dx)
        return residual

    @partial(jax.jit, static_argnums=(0,))
    def trim(
        self,
        t: float,
        X: Float[Array, "2"],
        V: Float[Array, "2"],
        aoa_prev_deg: float,
        engine_amps: float,
    ):
        """
        From the controller get all the variables that are marked as trim variables then solve the system of equations that ensure the forces normal to the trajectory are zero.

        Args:
            t (float): Time
            X (FloatArray): Position
            V (FloatArray): Velocity
            aoa_prev (float): Previous Angle of Attack in degrees
            engine_amps (float): Engine Amps
        """

        # Get Trajectory Information
        dh_dx: float = self.dy_dx(X[0])
        dh2_dx2: float = self.d2y_dx2(X[0])

        thrust = self.vehicle.motor.thrust(velocity=jnp.linalg.norm(V), current=engine_amps)

        x0 = jnp.atleast_1d(jnp.deg2rad(aoa_prev_deg))
        self.broyden.verbose = 0
        sol = self.broyden.run(init_params=x0, thrust=thrust, V=V, dh_dx=dh_dx, dh2_dx2=dh2_dx2)
        # jprint("sol : {}", sol)
        # sol = fsolve(f_to_trim, aoa_prev_deg, xtol=1e-09)

        aoa_new = jnp.atleast_1d(sol.params)
        amps_new = engine_amps
        # Print the residual of the function
        residual = self.f_to_trim(aoa_new, thrust, V, dh_dx, dh2_dx2)
        is_failed = residual > 1e-6
        is_failed = jnp.any(is_failed)

        def true_fun():
            jprint(
                "Trim Failed:\tTime: {}, Residual: {}, AoA: {}",
                t,
                self.f_to_trim(aoa_new, thrust, V, dh_dx, dh2_dx2),
                aoa_new,
            )
            # Print diagnostic information the args to the function
            jprint("\tArgs: thrust: {}, V: {}, dh_dx: {}, dh2_dx2: {}", thrust, V, dh_dx, dh2_dx2)
            jprint("\tx0: {}", x0)
            jprint("\tResidual: {}", residual)
            return jnp.atleast_1d(aoa_new)

        def false_fun():
            # jprint("Trim Successful:\tTime: {}, Residual: {}, AoA: {}", t, self.f_to_trim(aoa_new, thrust, V, dh_dx, dh2_dx2), aoa_new)
            return aoa_new

        # Use lax.cond to choose AoA based on residual
        aoa_new = jax.lax.cond(
            is_failed,
            true_fun,
            false_fun,
        )

        # Get the aerodynamic forces
        lift, drag, torque = self.vehicle.get_aerodynamic_forces(jnp.linalg.norm(V), aoa_new)
        return aoa_new, amps_new, lift, thrust, drag, torque

    @partial(jax.jit, static_argnums=(0,))
    def dvdt(
        self,
        t: float,
        X: Float[Array, "2"],
        V: Float[Array, "2"],
        prev_state: Float[Array, "2"],
    ) -> tuple[Float[Array, "2"], Float[Array, "8"]]:

        G: float = 9.81
        m: float = self.vehicle.mass

        # The carry of jax is
        aoa_prev, amps_prev = prev_state

        aoa, amps, lift, thrust, drag, torque = self.trim(t, X, V, aoa_prev, amps_prev)
        # Aoa is in degrees
        dh_dx = self.dy_dx(X[0])
        alpha = jnp.deg2rad(aoa) + jnp.arctan(dh_dx)

        thrust_x = thrust * jnp.cos(alpha)
        thrust_y = thrust * jnp.sin(alpha)

        acc_x = (
            thrust_x  # Thrust
            - lift * dh_dx / jnp.sqrt(dh_dx**2 + 1)  # Lift
            - drag / jnp.sqrt(dh_dx**2 + 1)  # Drag
            # +  Elevator drag due to control
        ) / m

        acc_y = (
            thrust_y  # Thrust
            + lift / jnp.sqrt(dh_dx**2 + 1)  # Lift
            - drag * dh_dx / jnp.sqrt(dh_dx**2 + 1)  # Drag
            - m * G  # Weight
            # + Elevator lift due to control
        ) / m

        F: Float[Array, "dim1"] = jnp.array([acc_x, acc_y], dtype=float).squeeze()

        state_now = jnp.array([aoa, amps])
        return F, state_now

    @partial(jax.jit, static_argnums=(0,))
    def timestep(
        self, t: float, 
        y: FloatArray, 
        *args
    ) -> FloatArray:
        x = y[:2]    
        v = y[2:]
        current_state = self.get_control()

        def normal():
            xdot = self.dxdt(x, v).reshape(-1)
            vdot, _ = self.dvdt(t, x, v, current_state)
            vdot = vdot.reshape(-1)
            return jnp.hstack([xdot, vdot])
        
        def problem():
            return jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan])
        
        def error_check(curr_y):
            def nan_check(curr_y):
                return jnp.isnan(curr_y).any()

            def crash_check(curr_y):
                return curr_y[1] < self.operating_floor

            def negative_v_check(curr_y):
                return curr_y[2] < 0

            def handle_error(_):
                # jprint("Error encountered at time: {}", t)
                # jprint("State: {}", curr_y)
                return True
            
            def ok(_):
                return False

            nan_result = lax.cond(nan_check(curr_y), handle_error, ok, None)
            crash_result = lax.cond(crash_check(curr_y), handle_error, ok, None)
            negative_v_result = lax.cond(negative_v_check(curr_y), handle_error, ok, None)

            return nan_result | crash_result | negative_v_result 

        # If there is a Nan Value in Y retrun error() else compute normal
        return jax.lax.cond(
            error_check(y),
            problem,
            normal,
        ) 

    def plot_history(self, axs: list[Axes] | None = None, fig=None) -> None:
        print(type(axs))

        if isinstance(axs, ndarray):
            axs = axs.tolist()
            # Get the figure
        # error
        if not isinstance(axs, list):
            fig = plt.figure()
            all_axs: list[Axes] = fig.subplots(3, 2, squeeze=False).flatten().tolist()
            axes_provided = False
        else:
            all_axs = axs
            axes_provided = True

        if fig is None:
            fig = all_axs[0].get_figure()
            if fig is None:
                raise ValueError("The figure is None")

        fig.suptitle(f"Trajectory: {self.name}")
        # Axes 0 is the Trajectory
        x = self.positions["x"]
        # Remove the nan values
        x_arr = jnp.array([i for i in x if i == i], dtype=float)
        h = self.y(x_arr)

        all_axs[0].plot(self.positions["x"], self.positions["y"], label="Trajectory", color="blue")
        all_axs[0].plot(x_arr, h, label="Function", color="Black", linestyle="--")
        all_axs[0].set_title("Trajectory")
        all_axs[0].set_xlabel("x")
        all_axs[0].set_ylabel("y")
        # On the same ax with different scale also plot the angle gamma
        ax2: Axes = all_axs[0].twinx()  # type: ignore
        color = "tab:red"
        ax2.plot(self.positions["x"], self.orientation["gamma"], label="Gamma", color=color)
        ax2.set_ylabel("Angle Gamma [rad]", color=color)  # we already handled the x-label with ax1
        ax2.tick_params(axis="y", labelcolor=color)

        # Axes 1 is the Velocity
        all_axs[1].plot(self.times, self.velocities["x"], label="Vx")
        all_axs[1].plot(self.times, self.velocities["y"], label="Vy")
        velocity_mag = [((vx**2 + vy**2) ** 0.5) for vx, vy in zip(self.velocities["x"], self.velocities["y"])]
        all_axs[1].plot(self.times, velocity_mag, label="V")
        all_axs[1].set_title("Velocity")
        all_axs[1].set_xlabel("Time")
        all_axs[1].set_ylabel("Velocity")

        # Axes 2 is the Forces
        all_axs[2].plot(self.times, self.forces["x"], label="Fx")
        all_axs[2].plot(self.times, self.forces["y"], label="Fy")
        all_axs[2].plot(self.times, self.forces["lift"], label="Lift")
        all_axs[2].plot(self.times, self.forces["thrust"], label="Thrust")
        all_axs[2].plot(self.times, self.forces["drag"], label="Drag")
        all_axs[2].plot(self.times, self.forces["torque"], label="Torque")
        all_axs[2].set_title("Forces")
        all_axs[2].set_xlabel("Time")

        # Axes 3 is the Control Inputs
        for key in self.control_inputs.keys():
            all_axs[3].plot(self.times, self.control_inputs[key], label=key)
        all_axs[3].set_title("Control Inputs")
        all_axs[3].set_xlabel("Time")

        # Axes 4 is the Angles
        all_axs[4].plot(self.times, self.orientation["alpha"], label="Alpha")
        all_axs[4].plot(self.times, self.orientation["aoa"], label="AoA")
        all_axs[4].set_title("Orientations")
        all_axs[4].set_xlabel("Time")

        # Axes 5 is the Accelerations
        all_axs[5].plot(self.times, self.accelerations["x"], label="Ax")
        all_axs[5].plot(self.times, self.accelerations["y"], label="Ay")
        all_axs[5].set_title("Accelerations")
        all_axs[5].set_xlabel("Time")

        # Clear the legends
        for ax in all_axs:
            try:
                # Clear the legends
                ax.get_legend().remove()
            except AttributeError:
                pass
            ax.legend()
            ax.grid()

        # Apply spacing to the plots
        fig.tight_layout()
        fig.canvas.flush_events()
        fig.canvas.draw()
        fig.show()
        import time

        time.sleep(0.1)

    # def to_nonlinear_sys(self) -> NonLinearSystem:
    #     # Define the timestep function for the system in jax pure functions
    #     import jax
    #     from jax import numpy as jnp

    #     def y(x):
    #         return self.fun(x)

    #     y = jax.jit(y)
    #     dy_dx = jax.grad(self.fun)
    #     dy_dx = jax.jit(dy_dx)

    #     d2y_dx2 = jax.grad(dy_dx)
    #     d2y_dx2 = jax.jit(d2y_dx2)

    #     def dxdt_jax(
    #         X: jnp.ndarray,
    #         V: jnp.ndarray,
    #     ) -> jnp.ndarray:
    #         DX: jnp.ndarray = V
    #         return DX

    #     def trim_jax(t, X, V, aoa_prev: float, engine_amps):

    #         G = 9.81

    #         # Get Trajectory Information
    #         dh_dx: float = dy_dx(X[0])
    #         dh2_dx2: float = d2y_dx2(X[0])

    #         thrust = self.vehicle.motor.thrust(velocity=jnp.linalg.norm(V), current=engine_amps)
    #         m = self.vehicle.mass

    #         # Get the aerodynamic forces
    #         def f(aoa):
    #             lift, _, _ = self.vehicle.get_aerodynamic_forces(float(np.linalg.norm(V)), aoa)
    #             alpha = aoa + dh_dx
    #             return (
    #                 thrust * (np.sin(alpha) - dh_dx * np.cos(alpha))
    #                 + lift * np.sqrt(1 + dh_dx**2)
    #                 - m * G
    #                 - m * dh2_dx2 * V[0] ** 2
    #             )

    #         # Import the fsolve function from scipy
    #         from jax.scipy.optimize import minimize

    #         aoa = minimize(fun=f, x0=np.deg2rad(aoa_prev), tol=1e-07, method="BFGS")[0]
    #         # Print the residual of the function
    #         # print(f"\tResidual: {f(aoa)}, AoA: {aoa}")
    #         aoa_deg = float(aoa)  # In radians

    #         # Get the aerodynamic forces
    #         lift, drag, torque = self.vehicle.get_aerodynamic_forces(float(np.linalg.norm(V)), aoa_deg)
    #         return aoa, lift, thrust, drag, torque

    #     def dvdt_jax(
    #         t: float,
    #         X: jnp.ndarray,
    #         V: jnp.ndarray,
    #     ) -> jnp.ndarray:

    #         G: float = 9.81
    #         m: float = self.vehicle.mass

    #         dh_dx = dy_dx(X[0])
    #         amps = 30
    #         aoa_prev = 0
    #         aoa, lift, thrust, drag, torque = trim_jax(t, X, V, aoa_prev, amps)
    #         alpha: float = np.deg2rad(aoa) + dh_dx

    #         thrust_x = thrust * np.cos(alpha)
    #         thrust_y = thrust * np.sin(alpha)

    #         acc_x: float = (
    #             thrust_x  # Thrust
    #             - lift * dh_dx / np.sqrt(dh_dx**2 + 1)  # Lift
    #             - drag / np.sqrt(dh_dx**2 + 1)  # Drag
    #             # +  Elevator drag due to control
    #         ) / m

    #         acc_y: float = (
    #             thrust_y  # Thrust
    #             + lift / np.sqrt(dh_dx**2 + 1)  # Lift
    #             - drag * dh_dx / np.sqrt(dh_dx**2 + 1)  # Drag
    #             - m * G  # Weight
    #             # + Elevator lift due to control
    #         ) / m

    #         F = jnp.hstack([acc_x, acc_y])
    #         return F

    #     def timestep(t: float, y: jnp.ndarray) -> jnp.ndarray:
    #         x = y[:2]
    #         v = y[-2:]

    #         # Check if the airplane is still in the air
    #         if x[1] < self.operating_floor:
    #             jprint("Crash")
    #             return jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan])
    #         if x[0] < 0:
    #             jprint("Negative x")
    #             return jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan])

    #         if v[0] < 0:
    #             jprint("Negative v")
    #             return jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan])

    #         xdot = dxdt_jax(x, v).reshape(-1)
    #         vdot = dvdt_jax(t, x, v).reshape(-1)

    #         if jnp.isnan(vdot).any():
    #             return jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan])

    #         return jnp.hstack([xdot, vdot])

    #     def f(t: float, y: jnp.ndarray) -> jnp.ndarray:
    #         return timestep(t, y)

    #     # The carry of jax should be done:
    #     # f(
    #     # t, y,
    #     # rest:
    #     # self.accelerations["x"].append(acc_x)
    #     # self.accelerations["y"].append(acc_y)
    #     # self.forces["x"].append(F[0])
    # self.forces["y"].append(F[1])
    # self.forces["lift"].append(lift)
    # self.forces["thrust"].append(thrust)
    # self.forces["drag"].append(drag)
    # self.forces["torque"].append(torque)
    # self.control_inputs["aoa"].append(aoa)
    # self.control_inputs["engine_amps"].append(amps)
    # self.orientation["alpha"].append(np.deg2rad(alpha))
    # self.orientation["aoa"].append(np.deg2rad(aoa))
    # self.orientation["gamma"].append(np.deg2rad(dh_dx))
    # )

    # f = jax.jit(f)
    # return NonLinearSystem(f)
