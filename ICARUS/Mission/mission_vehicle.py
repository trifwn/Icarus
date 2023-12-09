import numpy as np
from pandas import DataFrame

from ICARUS.Core.types import FloatArray
from ICARUS.Database import DB
from ICARUS.Mission.Trajectory.trajectory import Trajectory
from ICARUS.Propulsion.engine import Engine
from ICARUS.Vehicle.lifting_surface import Lifting_Surface
from ICARUS.Vehicle.plane import Airplane


class Mission_Vehicle:
    def __init__(self, airplane: Airplane, engine: Engine) -> None:
        self.airplane: Airplane = airplane
        self.motor: Engine = engine
        self.cldata = DB.vehicles_db.data[airplane.name]

        elevator: Lifting_Surface | None = None
        for surf in airplane.surfaces:
            if surf.name == "tail":
                self.l_e = surf.origin[0]
                elevator = surf
        if elevator is None:
            raise Exception("Elevator not found")

        self.elevator: Lifting_Surface = elevator
        self.elevator_max_deflection = 30
        self.Ixx: float = airplane.total_inertia[0]
        self.l_m: float = -0.4

    @staticmethod
    def interpolate_polars(aoa: float, cldata: DataFrame) -> tuple[FloatArray, FloatArray, FloatArray]:
        cl = np.interp(aoa, cldata["AoA"], cldata["GNVP7 Potential CL"])
        cd = np.interp(aoa, cldata["AoA"], cldata["GNVP7 Potential CD"])
        cm = np.interp(aoa, cldata["AoA"], cldata["GNVP7 Potential Cm"])
        return cl, cd, cm

    def get_elevator_lift_over_cl(
        self,
        velocity: float,
        aoa: float,
    ) -> float:
        aoa = np.rad2deg(aoa)
        density = 1.225
        lift_over_cl = np.pi * density * velocity**2 * self.elevator.S

        return lift_over_cl

    def get_lift_drag_torque(
        self,
        velocity: float,
        aoa: float,
    ):
        aoa = np.rad2deg(aoa)
        cl, cd, cm = self.interpolate_polars(
            aoa,
            self.cldata[
                [
                    "GNVP7 Potential CL",
                    "GNVP7 Potential CD",
                    "GNVP7 Potential Cm",
                    "AoA",
                ]
            ],
        )
        density = 1.225
        lift = cl * 0.5 * density * velocity**2 * self.airplane.S
        drag = cd * 0.5 * density * velocity**2 * self.airplane.S
        torque = cm * 0.5 * density * velocity**2 * self.airplane.S * self.airplane.mean_aerodynamic_chord

        return lift, drag, torque

    def dxdt(
        self,
        X: FloatArray,
        V: FloatArray,
    ) -> FloatArray:
        DX: FloatArray = V
        return DX

    def dvdt(self, X: FloatArray, V: FloatArray, trajectory: Trajectory, verbosity: int = 0):
        dh_dx: float = trajectory.first_derivative_x_fd(X[0])
        dh2_dx2: float = trajectory.second_derivative_x_fd(X[0])
        dh_3_dx3: float = trajectory.third_derivative_x_fd(X[0])

        # alpha = X[2]
        # aoa: float = X[2] - dh_dx
        aoa = 1
        alpha: float = aoa + dh_dx

        lift, drag, torque = self.get_lift_drag_torque(float(np.linalg.norm(V)), aoa)

        G: float = 9.81
        m: float = self.airplane.M

        # lift_el_over_cl = self.get_elevator_lift_over_cl(np.linalg.norm(V), aoa)
        # elevator_angle = 0

        # Calculate the thrust and elevator angle required to maintain course
        # aX = b , X = [thrust, elevator_angle]
        #
        # EQUATION OF MOTION IN X,H (basically projected to the trajectory)
        # DIRECTION TO MAINTAIN COURSE
        # a_11  = [(np.sin(alpha) - dh_dx * np.cos(alpha))]
        # a_12  = [lift_el_over_cl * 2pi * [dh_dx +1] / [np.sqrt(dh_dx**2 + 1)]]
        # b_1 = [m * G + m * dh2_dx2 * (V[0] **2) - lift * np.sqrt(dh_dx**2 + 1)]
        #
        # EQUATION OF ANGULAR MOMENTUM
        # a_21 = [0]
        # a_22 = [l_e * lift_el_over_cl * 2pi * [dh_dx +1] / [np.sqrt(dh_dx**2 + 1)] ]
        # b_2 = [torque]
        #
        # aX = b
        # X = a^-1 * b
        #

        # g1 = 1 / (np.sqrt(dh_dx**2 + 1))
        # g2 = - dh_dx / (np.sqrt(dh_dx**2 + 1)**2)

        # angular_acceleration = (
        #     g2 * (dh2_dx2 * V[0]) **2 +
        #     g1 +
        #    # dh2_dx2 * ormh_mass +
        #     dh_3_dx3 * V[0] **2
        # ) # NEED TO FIND THIS d^2 theta/dt^2 = d^2 arctan(h') /dt^2

        # a_11: float  = (np.sin(alpha) - dh_dx * np.cos(alpha))
        # a_12: float  = lift_el_over_cl  * (np.sqrt(dh_dx**2 + 1))
        # b_1 : float= m * G + m * dh2_dx2 * (V[0] **2) - lift * np.sqrt(dh_dx**2 + 1)
        # a_21: float = 0
        # a_22: float = - lift_el_over_cl  / (np.sqrt(dh_dx**2 + 1)) * l_e
        # b_2 : float= - torque + angular_acceleration * Ixx

        # a = np.array([[a_11, a_12], [a_21, a_22]])
        # b = np.array([b_1, b_2])
        # print(f"lift el over cl: {lift_el_over_cl}")
        # print(f'dh_dx: {dh_dx}')
        # print(f"\t{a_11} * x + {a_12} * y = {b_1}")
        # print(f"\t{a_21} * x + {a_22} * y = {b_2}")
        # print(a)
        # print(b)
        # sol = np.linalg.solve(a,b)
        # thrust = sol[0]
        # elevator_angle = sol[1]

        thrust: float = m * G + m * dh2_dx2 * V[0] ** 2 - lift * np.sqrt(dh_dx**2 + 1)
        thrust = thrust / (np.sin(alpha) - dh_dx * np.cos(alpha))

        # Check if thrust can  be provided in the motor dataset. It should be between the minimum and
        # maximum thrust. If not, the motor is not able to provide the thrust required to maintain
        # course.
        avail_thrust = self.motor.get_available_thrust(velocity=np.linalg.norm(V))
        if thrust < avail_thrust[0]:
            if verbosity:
                print(f"\tThrust too low {thrust}, {avail_thrust[1]}")
            # thrust = avail_thrust[0]

        if thrust > avail_thrust[1]:
            if verbosity:
                print(f"\tThrust too high {thrust},{avail_thrust[1]}")
            # thrust = avail_thrust[1]

        thrust_x = thrust * np.cos(alpha)
        thrust_y = thrust * np.sin(alpha)

        Fx: float = (
            1
            / m
            * (
                thrust_x  # Thrust
                - lift * dh_dx / np.sqrt(dh_dx**2 + 1)  # Lift
                - drag / np.sqrt(dh_dx**2 + 1)  # Drag
                # - 0* lift_el_over_cl * elevator_angle * dh_dx / (np.sqrt(dh_dx**2 + 1))  # Elevator drag due to control
            )
        )

        Fy: float = (
            1
            / m
            * (
                thrust_y  # Thrust
                + lift / np.sqrt(dh_dx**2 + 1)  # Lift
                - drag * dh_dx / np.sqrt(dh_dx**2 + 1)  # Drag
                # + 0* lift_el_over_cl * elevator_angle / (np.sqrt(dh_dx**2 + 1))  # Elevator lift due to control
                - m * G  # Weight
            )
        )
        # My = (
        #     dt
        #     / self.Ixx
        #     * (
        #         self.l_e * lift_el_over_cl * elevator_angle / (np.sqrt(dh_dx**2 + 1))
        #         - torque  # Elevator lift due to control  # Torque due to Wing
        #     )
        # )

        if verbosity > 1:
            print(f"\talpha: {np.rad2deg(alpha)}")
            print(f"\tThrust_x = {thrust_x}\n\tThrust_y = {thrust_y}")
            print(f"\tThrust: {thrust}\n\tLift = {lift}\n\tDrag = {drag}\n\tWeight = {m * G}\n\n")

        F: FloatArray = np.array([Fx, Fy])
        return F  # , thrust, elevator_angle
