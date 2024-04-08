from typing import Callable
from scipy.optimize import fsolve

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray


from ICARUS.core.types import AnyFloat
from ICARUS.core.types import FloatArray
from ICARUS.dynamical_systems.first_order_system import NonLinearSystem
from ICARUS.mission.mission_vehicle import MissionVehicle


class MissionTrajectory:
    def __init__(
        self,
        name: str,
        trajectory_function: Callable[[AnyFloat], FloatArray],
        vehicle: MissionVehicle,
        operating_floor: float,
        verbosity: int = 2,
    ) -> None:
        self.fun: Callable[[AnyFloat], FloatArray] = trajectory_function
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

    def clear_history(self):
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

    def y(self, x: float | FloatArray) -> FloatArray:
        return self.fun(x)

    def dy_dx_fd(self, x: float | FloatArray) -> FloatArray:
        h = 0.0001
        return (self.fun(x + h) - self.fun(x - h)) / (2 * h)

    def d2y_dx2_fd(self, x: float | FloatArray) -> FloatArray:
        # Second derivative
        h = 0.0001
        return (self.fun(x + h) - 2 * self.fun(x) + self.fun(x - h)) / (h**2)

    def dxdt(
        self,
        X: FloatArray,
        V: FloatArray,
    ) -> FloatArray:
        DX: FloatArray = V
        return DX

    def trim(self, t, X, V, aoa_prev: float, engine_amps):
        """
        From the controller get all the variables that are marked as trim variables then solve the system of equations that ensure the forces normal to the trajectory are zero.
        """
        G = 9.81

        # Get Trajectory Information
        dh_dx: float = float(self.dy_dx_fd(X[0]))
        dh2_dx2: float = float(self.d2y_dx2_fd(X[0]))

        thrust = self.vehicle.motor.thrust(
            velocity=float(np.linalg.norm(V)), current=engine_amps
        )
        m = self.vehicle.mass

        # Get the aerodynamic forces
        def f(aoa):
            lift, _, _ = self.vehicle.get_aerodynamic_forces(
                float(np.linalg.norm(V)), aoa
            )
            alpha = aoa + dh_dx
            return (
                thrust * (np.sin(alpha) - dh_dx * np.cos(alpha))
                + lift * np.sqrt(1 + dh_dx**2)
                - m * G
                - m * dh2_dx2 * V[0] ** 2
            )


        aoa = fsolve(f, np.deg2rad(aoa_prev), xtol=1e-07)[0]
        # Print the residual of the function
        # print(f"\tResidual: {f(aoa)}, AoA: {aoa}")
        aoa_deg = np.rad2deg(aoa)

        # Get the aerodynamic forces
        lift, drag, torque = self.vehicle.get_aerodynamic_forces(
            float(np.linalg.norm(V)), aoa_deg
        )
        return aoa, lift, thrust, drag, torque

    def dvdt(
        self,
        t: float,
        X: FloatArray,
        V: FloatArray,
    ) -> FloatArray:

        G: float = 9.81
        m: float = self.vehicle.mass

        try:
            aoa_prev = self.control_inputs["aoa"][-1]
            amps = self.control_inputs["engine_amps"][-1]
        except:
            aoa_prev = 0
            amps = 30

        dh_dx = float(self.dy_dx_fd(X[0]))
        aoa, lift, thrust, drag, torque = self.trim(t, X, V, aoa_prev, amps)
        alpha: float = np.deg2rad(aoa) + dh_dx

        thrust_x = thrust * np.cos(alpha)
        thrust_y = thrust * np.sin(alpha)

        acc_x: float = (
            thrust_x  # Thrust
            - lift * dh_dx / np.sqrt(dh_dx**2 + 1)  # Lift
            - drag / np.sqrt(dh_dx**2 + 1)  # Drag
            # +  Elevator drag due to control
        ) / m

        acc_y: float = (
            thrust_y  # Thrust
            + lift / np.sqrt(dh_dx**2 + 1)  # Lift
            - drag * dh_dx / np.sqrt(dh_dx**2 + 1)  # Drag
            - m * G  # Weight
            # + Elevator lift due to control
        ) / m

        if self.verbosity > 1:
            print(f"\tTime: {t}")
            print(f"\tPosition: {X}")
            print(f"\tVelocity: {V}")
            print(f"\tGamma = {np.rad2deg(dh_dx)}")
            print(f"\tVelocity: {np.linalg.norm(V)}")
            print(f"\tAoA: {np.rad2deg(aoa)}")
            print(f"\talpha: {np.rad2deg(alpha)}")
            print(f"\tThrust_x = {thrust_x}\n\tThrust_y = {thrust_y}")
            print(f"\tThrust: {thrust}")
            print(f"\tDrag = {drag}")
            print(f"\tLift = {lift}")
            print(f"\tWeight = {m * G}\n\n")

        F: FloatArray = np.array([acc_x, acc_y], dtype=float)

        step_successful = np.isnan(F).any() == False
        if step_successful:
            # Register the state
            self.times.append(t)
            self.positions["x"].append(X[0])
            self.positions["y"].append(X[1])

            self.velocities["x"].append(V[0])
            self.velocities["y"].append(V[1])

            self.accelerations["x"].append(acc_x)
            self.accelerations["y"].append(acc_y)

            self.forces["x"].append(F[0])
            self.forces["y"].append(F[1])

            self.forces["lift"].append(lift)
            self.forces["thrust"].append(thrust)
            self.forces["drag"].append(drag)
            self.forces["torque"].append(torque)

            self.control_inputs["aoa"].append(aoa)
            self.control_inputs["engine_amps"].append(amps)

            self.orientation["alpha"].append(np.deg2rad(alpha))
            self.orientation["aoa"].append(np.deg2rad(aoa))
            self.orientation["gamma"].append(np.deg2rad(dh_dx))

        return F

    def timestep(self, t: float, y: FloatArray) -> FloatArray:
        x = y[:2]
        v = y[-2:]

        # Check if the airplane is still in the air
        if x[1] < self.operating_floor:
            print("Crash")
            return np.array([np.nan, np.nan, np.nan, np.nan])
        if x[0] < 0:
            print("Negative x")
            return np.array([np.nan, np.nan, np.nan, np.nan])

        if v[0] < 0:
            print("Negative v")
            return np.array([np.nan, np.nan, np.nan, np.nan])

        xdot = self.dxdt(x, v).reshape(-1)
        vdot = self.dvdt(t, x, v).reshape(-1)

        if np.isnan(vdot).any():
            return np.array([np.nan, np.nan, np.nan, np.nan])

        return np.hstack([xdot, vdot])

    def plot_history(self, axs: list[Axes] | None = None) -> None:
        if isinstance(axs, ndarray):
            axs = axs.tolist()

        if not isinstance(axs, list):
            fig = plt.figure()
            all_axs: list[Axes] = fig.subplots(3, 2, squeeze=False).flatten().tolist()

        # Axes 0 is the Trajectory
        x = self.positions["x"]
        # Remove the nan values
        x_arr = np.array([i for i in x if i == i], dtype=float)
        h = self.y(x_arr)

        all_axs[0].plot(
            self.positions["x"], self.positions["y"], label="Trajectory", color="blue"
        )
        all_axs[0].plot(x_arr, h, label="Function", color="Black", linestyle="--")
        all_axs[0].set_title("Trajectory")
        all_axs[0].set_xlabel("x")
        all_axs[0].set_ylabel("y")
        all_axs[0].legend()
        # On the same ax with different scale also plot the angle gamma
        ax2: Axes = all_axs[0].twinx()  # type: ignore
        color = "tab:red"
        ax2.plot(
            self.positions["x"], self.orientation["gamma"], label="Gamma", color=color
        )
        ax2.set_ylabel(
            "Angle Gamma [rad]", color=color
        )  # we already handled the x-label with ax1
        ax2.tick_params(axis="y", labelcolor=color)

        # Axes 1 is the Velocity
        all_axs[1].plot(self.times, self.velocities["x"], label="Vx")
        all_axs[1].plot(self.times, self.velocities["y"], label="Vy")
        velocity_mag = [
            ((vx**2 + vy**2) ** 0.5)
            for vx, vy in zip(self.velocities["x"], self.velocities["y"])
        ]
        all_axs[1].plot(self.times, velocity_mag, label="V")
        all_axs[1].set_title("Velocity")
        all_axs[1].set_xlabel("Time")
        all_axs[1].set_ylabel("Velocity")
        all_axs[1].legend()

        # Axes 2 is the Forces
        all_axs[2].plot(self.times, self.forces["x"], label="Fx")
        all_axs[2].plot(self.times, self.forces["y"], label="Fy")
        all_axs[2].plot(self.times, self.forces["lift"], label="Lift")
        all_axs[2].plot(self.times, self.forces["thrust"], label="Thrust")
        all_axs[2].plot(self.times, self.forces["drag"], label="Drag")
        all_axs[2].plot(self.times, self.forces["torque"], label="Torque")
        all_axs[2].set_title("Forces")
        all_axs[2].set_xlabel("Time")
        all_axs[2].legend()

        # Axes 3 is the Control Inputs
        for key in self.control_inputs.keys():
            all_axs[3].plot(self.times, self.control_inputs[key], label=key)
        all_axs[3].set_title("Control Inputs")
        all_axs[3].set_xlabel("Time")
        all_axs[3].legend()

        # Axes 4 is the Angles
        all_axs[4].plot(self.times, self.orientation["alpha"], label="Alpha")
        all_axs[4].plot(self.times, self.orientation["aoa"], label="AoA")
        all_axs[4].set_title("Orientations")
        all_axs[4].set_xlabel("Time")
        all_axs[4].legend()

        # Axes 5 is the Accelerations
        all_axs[5].plot(self.times, self.accelerations["x"], label="Ax")
        all_axs[5].plot(self.times, self.accelerations["y"], label="Ay")
        all_axs[5].set_title("Accelerations")
        all_axs[5].set_xlabel("Time")
        all_axs[5].legend()

        # Apply spacing to the plots
        fig.tight_layout()

        fig.show()

    def to_nonlinear_sys(self) -> NonLinearSystem:
        # Define the timestep function for the system in jax pure functions
        import jax
        from jax import numpy as jnp
        from jax.debug import print as jprint

        def dxdt_jax(
            X: jnp.ndarray,
            V: jnp.ndarray,
        ) -> jnp.ndarray:
            DX: jnp.ndarray = V
            return DX

        def trim_jax(
            t, 
            X, 
            V, 
            aoa_prev: float, 
            engine_amps
        ):

            G = 9.81

            # Get Trajectory Information
            dh_dx: float = float(self.dy_dx_fd(X[0]))
            dh2_dx2: float = float(self.d2y_dx2_fd(X[0]))

            thrust = self.vehicle.motor.thrust(
                velocity=float(np.linalg.norm(V)), current=engine_amps
            )
            m = self.vehicle.mass

            # Get the aerodynamic forces
            def f(aoa):
                lift, _, _ = self.vehicle.get_aerodynamic_forces(
                    float(np.linalg.norm(V)), aoa
                )
                alpha = aoa + dh_dx
                return (
                    thrust * (np.sin(alpha) - dh_dx * np.cos(alpha))
                    + lift * np.sqrt(1 + dh_dx**2)
                    - m * G
                    - m * dh2_dx2 * V[0] ** 2
                )

            # Import the fsolve function from scipy
            from jax.scipy.optimize import minimize

            aoa = minimize(fun=f, x0=np.deg2rad(aoa_prev), tol=1e-07, method="BFGS")[0]
            # Print the residual of the function
            # print(f"\tResidual: {f(aoa)}, AoA: {aoa}")
            aoa_deg = np.rad2deg(aoa)

            # Get the aerodynamic forces
            lift, drag, torque = self.vehicle.get_aerodynamic_forces(
                float(np.linalg.norm(V)), aoa_deg
            )
            return aoa, lift, thrust, drag, torque
    
        def dvdt_jax(
            t: float,
            X: jnp.ndarray,
            V: jnp.ndarray,
        ) -> jnp.ndarray:

            G: float = 9.81
            m: float = self.vehicle.mass

            try:
                aoa_prev = self.control_inputs["aoa"][-1]
                amps = self.control_inputs["engine_amps"][-1]
            except:
                aoa_prev = 0
                amps = 30

            dh_dx = self.dy_dx_fd(X[0])
            aoa, lift, thrust, drag, torque = self.trim(t, X, V, aoa_prev, amps)
            alpha: float = np.deg2rad(aoa) + dh_dx

            thrust_x = thrust * np.cos(alpha)
            thrust_y = thrust * np.sin(alpha)

            acc_x: float = (
                thrust_x  # Thrust
                - lift * dh_dx / np.sqrt(dh_dx**2 + 1)  # Lift
                - drag / np.sqrt(dh_dx**2 + 1)  # Drag
                # +  Elevator drag due to control
            ) / m

            acc_y: float = (
                thrust_y  # Thrust
                + lift / np.sqrt(dh_dx**2 + 1)  # Lift
                - drag * dh_dx / np.sqrt(dh_dx**2 + 1)  # Drag
                - m * G  # Weight
                # + Elevator lift due to control
            ) / m

            F = jnp.hstack([acc_x, acc_y])

            step_successful = np.isnan(F).any() == False
            # Register the state
            # self.times.append(t)
            # self.positions["x"].append(X[0])
            # self.positions["y"].append(X[1])

            # self.velocities["x"].append(V[0])
            # self.velocities["y"].append(V[1])

            # self.accelerations["x"].append(acc_x)
            # self.accelerations["y"].append(acc_y)

            # self.forces["x"].append(F[0])
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

            return F

        def timestep(t: float, y: jnp.ndarray) -> jnp.ndarray:
            x = y[:2]
            v = y[-2:]

            # Check if the airplane is still in the air
            if x[1] < self.operating_floor:
                jprint("Crash")
                return jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan])
            if x[0] < 0:
                jprint("Negative x")
                return jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan])

            if v[0] < 0:
                jprint("Negative v")
                return jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan])

            xdot = dxdt_jax(x, v).reshape(-1)
            vdot = dvdt_jax(t, x, v).reshape(-1)

            if jnp.isnan(vdot).any():
                return jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan])

            return jnp.hstack([xdot, vdot])

        def f(t: float, y: jnp.ndarray) -> jnp.ndarray:
            return timestep(t, y)

        f = jax.jit(f)
        return NonLinearSystem(f)
