from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ICARUS.flight_dynamics import State


def eigenvalue_analysis(
    state_space_obj: LateralStateSpace | LongitudalStateSpace,
) -> None:
    # Compute Eigenvalues and Eigenvectors
    eigval_long, eigvec_long = np.linalg.eig(state_space_obj.A)
    state_space_obj.eigenvalues = eigval_long
    state_space_obj.eigenvectors = eigvec_long


class LateralStateSpace:
    def __init__(
        self,
        state: State,
        Y: dict[str, float],
        L: dict[str, float],
        N: dict[str, float],
    ):
        self.name = "Lateral"
        self.Y = Y
        self.L = L
        self.N = N

        mass: float = state.mass
        U: float = state.trim["U"]
        theta: float = state.trim["AoA"] * np.pi / 180
        u_e: float = U * np.cos(theta)
        w_e: float = U * np.sin(theta)
        G: float = state.environment.GRAVITY

        Ix, Iy, Iz, Ixz, Ixy, Iyz = state.inertia

        yv: float = Y["v"] / mass
        yp: float = (Y["p"] + mass * w_e) / mass
        yr: float = (Y["r"] - mass * u_e) / mass
        yphi: float = G * np.cos(theta)

        lv: float = (Iz * L["v"] + Ixz * N["v"]) / (Ix * Iz - Ixz**2)
        lp: float = (Iz * L["p"] + Ixz * N["p"]) / (Ix * Iz - Ixz**2)
        lr: float = (Iz * L["r"] + Ixz * N["r"]) / (Ix * Iz - Ixz**2)
        lphi: float = 0

        nv: float = (Ix * N["v"] + Ixz * L["v"]) / (Ix * Iz - Ixz**2)
        n_p: float = (Ix * N["p"] + Ixz * L["p"]) / (Ix * Iz - Ixz**2)
        nr: float = (Ix * N["r"] + Ixz * L["r"]) / (Ix * Iz - Ixz**2)
        nph: float = 0

        self.A = np.array(
            [
                [yv, yp, yr, yphi],
                [lv, lp, lr, lphi],
                [nv, n_p, nr, nph],
                [
                    0,
                    1,
                    0 * np.tan(theta),
                    0,
                ],  # Etkins uses np.tan(theta) here where as most writers use 0
            ],
        )

        self.A_DS = np.array(
            [
                [Y["v"], Y["p"], Y["r"], Y["phi"]],
                [L["v"], L["p"], L["r"], L["phi"]],
                [N["v"], N["p"], N["r"], N["phi"]],
                [0, 1, 0, 0],
            ],
        )

        self.eigenvalues = np.empty((4,), dtype=float)
        self.eigenvectors = np.empty((4, 4), dtype=float)
        self.dutch_roll: complex = 0 + 0j
        self.coupled_mode: complex = 0 + 0j
        self.roll_subsidance: float = 0.0
        self.spiral: float = 0.0
        self.n_modes = 3
        eigenvalue_analysis(self)
        # self.classify_modes()

    def classify_modes(self) -> None:
        """Classify the modes based on the eigenvalues. On the lateral case, the modes are:
        - Roll (1 mode)
        - Spiral (1 mode)
        - Dutch Roll (2 modes)
        There is the case where the roll and spiral mode become coupled and form a pair of
        2 conjugate value
        """
        # Get the number of modes by setting the imaginary part to abs
        eigen_values = set(
            np.array([val.real + 1j * np.abs(val.imag) for val in self.eigenvalues]),
        )
        n_modes = len(eigen_values)
        self.n_modes = n_modes

        if n_modes == 2:
            # Then we a dutch mode and a coupled roll-spiral mode
            # The dutch-roll mode is the one with the smallest complex part
            dutch_roll = min(eigen_values, key=lambda x: x.imag)
            coupled_mode = max(eigen_values, key=lambda x: x.imag)

            self.dutch_roll = dutch_roll
            self.coupled_mode = coupled_mode
            # NON EXISTENT
            self.roll_subsidance = np.nan
            self.spiral = np.nan
        elif n_modes == 3:
            # Then we have a dutch roll, a roll and a spiral mode
            # The complex eigenvalue is the dutch roll
            dutch_roll = max(eigen_values, key=lambda x: np.abs(x.imag))
            # The two real eigenvalues are the roll and spiral modes
            # The spiral mode is the one with the smallest real part
            spiral = min(
                [eigen_value for eigen_value in eigen_values if eigen_value.imag == 0],
                key=lambda x: x.real,
            )
            # The roll mode is the one with the largest real part
            roll_subsidance = max(
                [val for val in eigen_values if val.imag == 0],
                key=lambda x: x.real,
            )
            self.dutch_roll = dutch_roll
            self.roll_subsidance = roll_subsidance.real
            self.spiral = spiral.real
            # NON EXISTENT
            self.coupled_mode = np.nan
        else:
            # I have never seen this case so let's print the eigenvalues
            # And throw an error
            if n_modes == 4:
                print(
                    "4 distinct modes found in the lateral dynamics. (Overdamped Dutch Roll)",
                )
            else:
                print("1 mode found in the lateral dynamics")
            print(self.eigenvalues)
            raise ValueError

    def print_derivatives(
        self,
    ) -> None:
        print("Lateral Derivatives")
        print(f"Yv=\t{self.Y['v']}")
        print(f"Yp=\t{self.Y['p']}")
        print(f"Yr=\t{self.Y['r']}")
        print(f"Lv=\t{self.L['v']}")
        print(f"Lp=\t{self.L['p']}")
        print(f"Lr=\t{self.L['r']}")
        print(f"Nv=\t{self.N['v']}")
        print(f"Np=\t{self.N['p']}")
        print(f"Nr=\t{self.N['r']}")


class LongitudalStateSpace:
    def __init__(
        self,
        state: State,
        X: dict[str, float],
        Z: dict[str, float],
        M: dict[str, float],
    ):
        self.name = "Longitudal"
        self.X: dict[str, float] = X
        self.Z: dict[str, float] = Z
        self.M: dict[str, float] = M

        m: float = state.mass
        trim_velocity: float = state.trim["U"]
        theta: float = state.trim["AoA"] * np.pi / 180
        u_e: float = trim_velocity * np.cos(theta)
        w_e: float = trim_velocity * np.sin(theta)

        G: float = 9.81
        Ix, Iy, Iz, Ixz, Ixy, Iyz = state.inertia

        xu = X["u"] / m  # + (X["w_dot"] * Z["u"])/(m*(M-Z["w_dot"]))
        xw = X["w"] / m  # + (X["w_dot"] * Z["w"])/(m*(M-Z["w_dot"]))
        xq = (X["q"] - m * w_e) / (m)
        xth = -G * np.cos(theta)

        # xq += (X["w_dot"] * (Z["q"] + m * Ue))/(m*(m-Z["w_dot"]))
        # xth += - (X["w_dot"]*G * np.sin(theta))/((m-Z["w_dot"]))

        zu = Z["u"] / (m - Z["w_dot"])
        zw = Z["w"] / (m - Z["w_dot"])
        zq = (Z["q"] + m * u_e) / (m - Z["w_dot"])
        zth = -(m * G * np.sin(theta)) / (m - Z["w_dot"])

        mu = (M["u"] + Z["u"] * M["w_dot"] / (m - Z["w_dot"])) / Iy
        mw = (M["w"] + Z["w"] * M["w_dot"] / (m - Z["w_dot"])) / Iy
        mq = (M["q"] + M["w_dot"] * (Z["q"] + m * u_e) / (m - Z["w_dot"])) / Iy
        mth = -(m * G * np.sin(theta) * M["w_dot"]) / (Iy * (m - Z["w_dot"]))

        self.A_DS = np.array(
            [
                [X["u"], X["w"], X["q"], X["theta"]],
                [Z["u"], Z["w"], Z["q"], Z["theta"]],
                [M["u"], M["w"], M["q"], M["theta"]],
                [0, 0, 1, 0],
            ],
        )

        self.A = np.array(
            [[xu, xw, xq, xth], [zu, zw, zq, zth], [mu, mw, mq, mth], [0, 0, 1, 0]],
        )

        self.eigenvalues = np.empty((4,), dtype=complex)
        self.eigenvectors = np.empty((4, 4), dtype=float)
        self.short_period: complex = 0 + 0j
        self.phugoid: complex = 0 + 0j
        self.overdamped_short_period: list[float] = [0.0, 0.0]
        self.n_modes: int = 2
        eigenvalue_analysis(self)
        # self.classify_modes()

    def classify_modes(self) -> None:
        """Classify the modes based on the eigenvalues. On the longitudinal case, the modes are:
        - Phugoid  (2 modes)
        - Short Period (2 modes)
        The short period mode is oscillatory but can degenerate to two real roots
        """
        # Get the number of modes by setting the imaginary part to abs
        eigen_values = set(
            np.array([val.real + 1j * np.abs(val.imag) for val in self.eigenvalues]),
        )
        n_modes = len(eigen_values)
        self.n_modes = n_modes

        if n_modes == 2:
            # Then we have a phugoid and a short period mode
            # This is the mode with the smallest complex part
            phugoid = min(eigen_values, key=lambda x: x.imag)
            short_period = max(eigen_values, key=lambda x: x.imag)

            self.phugoid = phugoid
            self.short_period = short_period
        elif n_modes == 3:
            # Then we have a phugoid and two short period modes
            # The complex eigenvalue is the phugoid
            phugoid = max(eigen_values, key=lambda x: x.imag)
            # The two real eigenvalues are the short period modes
            short_period_modes = [
                eigen_val for eigen_val in eigen_values if eigen_val.imag == 0
            ]

            self.phugoid = phugoid
            self.short_period = np.nan
            self.overdamped_short_period = [val.real for val in short_period_modes]
        else:
            # I have never seen this case so let's print the eigenvalues
            # And throw an error
            if n_modes == 4:
                print("4 distinct modes found in the longitudinal dynamics")
            else:
                print("1 mode found in the longitudinal dynamics")
            print(self.eigenvalues)
            raise ValueError

    def print_derivatives(
        self,
    ) -> None:
        print("Longitudal Derivatives")
        print(f"Xu=\t{self.X['u']}")
        print(f"Xw=\t{self.X['w']}")
        print(f"Xq=\t{self.X['q']}")
        print(f"Zu=\t{self.Z['u']}")
        print(f"Zw=\t{self.Z['w']}")
        print(f"Zq=\t{self.Z['q']}")
        print(f"Mu=\t{self.M['u']}")
        print(f"Mw=\t{self.M['w']}")
        print(f"Mq=\t{self.M['q']}")


class StateSpace:
    def __init__(
        self,
        longitudal_state_space: LongitudalStateSpace,
        lateral_state_space: LateralStateSpace,
    ):
        self.longitudal = longitudal_state_space
        self.lateral = lateral_state_space
