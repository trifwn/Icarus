from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ICARUS.Flight_Dynamics.state import State


def eigenvalue_analysis(
    state_space_obj: "LateralStateSpace | LongitudalStateSpace",
) -> None:
    # Compute Eigenvalues and Eigenvectors
    eigvalLong, eigvecLong = np.linalg.eig(state_space_obj.A)
    state_space_obj.eigenvalues = eigvalLong
    state_space_obj.eigenvectors = eigvecLong


class LateralStateSpace:
    def __init__(
        self,
        state: "State",
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
        G: float = state.environment.GRAVITY

        Ix, Iy, Iz, Ixz, Ixy, Iyz = state.inertia

        yv: float = Y["v"] / mass
        yp: float = (Y["p"] + mass * U * np.sin(theta)) / mass
        yr: float = (Y["r"] - mass * U * np.cos(theta)) / mass
        yphi: float = -G * np.cos(theta)

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
                [0, 1, np.tan(theta), 0],
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
        eigenvalue_analysis(self)
        self.omegas = np.abs(self.eigenvalues)
        self.zetas = -(self.eigenvalues.real) / (self.omegas)

    def print_lateral_derivatives(
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
        state: "State",
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
        u_e: float = np.abs(trim_velocity * np.cos(theta))
        w_e: float = np.abs(trim_velocity * np.sin(theta))

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
        eigenvalue_analysis(self)
        self.omegas = np.abs(self.eigenvalues)
        self.zetas = -(self.eigenvalues.real) / (self.omegas)

    def print_lateral_derivatives(
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
