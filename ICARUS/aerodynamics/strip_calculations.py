import jax.numpy as jnp

from ICARUS.airfoils.airfoil import Airfoil


def get_gamma_distribution(
    self,
) -> None:
    if self.a_np is None:
        raise ValueError("You must solve the wing panels first")
    gammas = jnp.linalg.solve(self.a_np, self.RHS_np)
    self.w = jnp.matmul(self.b_np, gammas)

    self.gammas_mat = jnp.zeros((self.N - 1, self.M))
    self.w_mat = jnp.zeros((self.N - 1, self.M))

    for i in jnp.arange(0, (self.N - 1) * (self.M)):
        lp, kp = divmod(i, (self.M))
        self.gammas_mat[lp, kp] = gammas[i]
        self.w_mat[lp, kp] = self.w[i]
    self.calculate_strip_induced_velocities()


def get_aerodynamic_loads(self, umag: float, verbose: bool = False):
    if self.gammas_mat is None:
        self.get_gamma_distribution()

    L_pan = jnp.zeros((self.N - 1, self.M))
    D_pan = jnp.zeros((self.N - 1, self.M))
    D_trefftz = 0.0
    for i in jnp.arange(0, self.N - 1):
        for j in jnp.arange(0, self.M):
            dy: float = self.grid[i + 1, j, 1] - self.grid[i, j, 1]
            if j == 0:
                g = gammas_mat[i, j]
            else:
                g = gammas_mat[i, j] - gammas_mat[i, j - 1]
            L_pan[i, j] = self.dens * umag * dy * g
            D_pan[i, j] = -self.dens * dy * g * self.w_mat[i, j]

            if j == self.M - 2:
                D_trefftz += -self.dens / 2 * dy * gammas_mat[i, j] * self.w_mat[i, j]

    # Calculate the torque. The torque is calculated w.r.t. the CG
    # and is the sum of the torques of each panel times the distance
    # from the CG to the control point of each panel
    M = jnp.array([0, 0, 0], dtype=float)
    for i in jnp.arange(0, self.N - 1):
        for j in jnp.arange(0, self.M - 1):
            M += L_pan[i, j] * jnp.cross(self.control_points[i, j, :] - self.cog, self.control_nj[i, j, :])
            M += D_pan[i, j] * jnp.cross(self.control_points[i, j, :] - self.cog, self.control_nj[i, j, :])
    Mx, My, Mz = M

    D_pan = D_pan
    L_pan = L_pan
    L: float = float(jnp.sum(L_pan))
    D: float = D_trefftz  # np.sum(D_pan)
    D2: float = float(jnp.sum(D_pan))
    Mx: float = Mx
    My: float = My
    Mz: float = Mz

    CL: float = 2 * self.L / (self.dens * (umag**2) * self.S)
    CD: float = 2 * self.D / (self.dens * (umag**2) * self.S)
    Cm: float = 2 * self.My / (self.dens * (umag**2) * self.S * self.MAC)

    interpolation_success: bool = True
    try:
        integrate_polars_from_reynolds(umag)
    except ValueError as e:
        print("\tCould not interpolate polars! Got error:")
        print(f"\t{e}")
        interpolation_success = False

    if verbose:
        print(f"- Angle {self.alpha * 180 /jnp.pi}")
        print(f"\t--Using no penetration condition:")
        print(f"\t\tL:{self.L}\t|\tD (Trefftz Plane):{self.D}\tD2:{self.D2}\t|\tMy:{self.My}")
        print(f"\t\tCL:{self.CL}\t|\tCD_ind:{self.CD}\t|\tCm:{self.Cm}")
        if interpolation_success:
            print(f"\t--Using 2D polars:")
            print(f"\t\tL:{self.L_2D}\t|\tD:{self.D_2D}\t|\tMy:{self.My_2D}")
            print(f"\t\tCL:{self.CL_2D}\t|\tCD:{self.CD_2D}\t|\tCm:{self.Cm_2D}")

    return L, D, D2, Mx, My, Mz, CL, CD, Cm, L_pan, D_pan


def calculate_strip_induced_velocities(self) -> None:
    if self.w_mat is None:
        self.get_gamma_distribution()
    self.w_induced_strips = jnp.mean(self.w_mat, axis=1)


def calculate_strip_gamma(self) -> None:
    if self.gammas_mat is None:
        self.get_gamma_distribution()
    self.gamma_strips = jnp.mean(self.gammas_mat, axis=1)


def calc_strip_reynolds(self, umag: float) -> None:
    if self.w_induced_strips is None:
        self.calculate_strip_induced_velocities()
    self.calc_strip_chords()

    # Get the effective angle of attack of each strip
    self.strip_effective_aoa = jnp.arctan(self.w_induced_strips / umag) * 180 / jnp.pi + self.alpha * 180 / jnp.pi

    # Get the reynolds number of each strip
    strip_vel = jnp.sqrt(self.w_induced_strips**2 + umag**2)
    self.strip_reynolds = self.dens * strip_vel * self.chords / self.visc

    # Scan all wing segments and get the orientation of each airfoil
    # Match that orientation with the each strip and get the effective aoa
    # That is the angle of attack that the airfoil sees
    self.strip_airfoil_effective_aoa = jnp.zeros(self.N - 1)
    N: int = 0
    for wing_seg in self.wing_segments:
        for j in jnp.arange(0, wing_seg.N - 1):
            self.strip_airfoil_effective_aoa[N + j] = self.strip_effective_aoa[N + j] + wing_seg.orientation[0]
        N += wing_seg.N - 1


def integrate_polars_from_reynolds(self, uinf: float, solver: str = "Xfoil") -> None:
    # self.get_strip_reynolds(20, 1.225, 1.7894e-5)
    # if self.strip_reynolds is None:
    self.strip_CL_2D = jnp.zeros(self.N - 1)
    self.strip_CD_2D = jnp.zeros(self.N - 1)
    self.strip_Cm_2D = jnp.zeros(self.N - 1)

    self.L_2D: float | None = None
    self.D_2D: float | None = None
    self.My_2D: float | None = None

    self.CL_2D: float | None = None
    self.CD_2D: float | None = None
    self.Cm_2D: float | None = None

    self.calc_strip_reynolds(uinf)

    N: int = 0
    L: float = 0
    D: float = 0
    My_at_quarter_chord: float = 0
    for wing_seg in self.wing_segments:
        airfoil: Airfoil = wing_seg.root_airfoil
        for j in jnp.arange(0, wing_seg.N - 1):
            dy: float = float(jnp.mean(self.grid[N + j + 1, :, 1] - self.grid[N + j, :, 1]))

            CL, CD, Cm = DB.foils_db.interpolate_polars(
                reynolds=float(self.strip_reynolds[N + j]),
                airfoil_name=airfoil.name,
                aoa=float(self.strip_airfoil_effective_aoa[N + j]),
                solver=solver,
            )
            self.strip_CL_2D[N + j] = CL
            self.strip_CD_2D[N + j] = CD
            self.strip_Cm_2D[N + j] = Cm

            surface: float = float(self.chords[N + j]) * dy
            vel_mag: float = float(jnp.sqrt(self.w_induced_strips[N + j] ** 2 + uinf**2))
            dynamic_pressure: float = 0.5 * self.dens * vel_mag**2

            # "Integrate" the CL and CD of each strip to get the total L, D and My
            L += CL * surface * dynamic_pressure
            D += CD * surface * dynamic_pressure
            My_at_quarter_chord += Cm * surface * dynamic_pressure * float(self.chords[N + j])

        N += wing_seg.N - 1

    self.L_2D = L
    self.D_2D = D

    # Calculate Total Moment moving the moment from the quarter chord
    # to the cg and then add the moment of the lift and drag

    self.My_2D = My_at_quarter_chord - D * self.cog[0] + L * self.cog[0]

    self.CL_2D = 2 * self.L_2D / (self.dens * (uinf**2) * self.S)
    self.CD_2D = 2 * self.D_2D / (self.dens * (uinf**2) * self.S)
    self.Cm_2D = 2 * self.My_2D / (self.dens * (uinf**2) * self.S * self.MAC)
