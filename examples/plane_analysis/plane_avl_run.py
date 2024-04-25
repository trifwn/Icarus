import os

import numpy as np
from Planes.hermes import hermes

from ICARUS.computation.solvers.AVL.analyses.pertrubations import (
    avl_dynamic_analysis_fd,
)
from ICARUS.computation.solvers.AVL.analyses.pertrubations import (
    avl_dynamic_analysis_implicit,
)
from ICARUS.computation.solvers.AVL.analyses.pertrubations import process_avl_fd_res
from ICARUS.computation.solvers.AVL.analyses.pertrubations import process_avl_impl_res
from ICARUS.computation.solvers.AVL.analyses.polars import avl_angle_run
from ICARUS.computation.solvers.AVL.analyses.polars import process_avl_angles_run
from ICARUS.computation.solvers.XFLR5.polars import read_polars_3d
from ICARUS.database import EXTERNAL_DB
from ICARUS.environment.definition import EARTH_ISA
from ICARUS.flight_dynamics.state import State

plane = hermes("hermes")

env = EARTH_ISA
UINF = 20
solver2D = "Xfoil"
state = State(name="Unstick", airplane=plane, environment=EARTH_ISA, u_freestream=UINF)

angles = np.linspace(-10, 10, 11)

avl_angle_run(plane, state, solver2D, angles)
pol_df = process_avl_angles_run(plane, state, angles)

from ICARUS.database import EXTERNAL_DB

XFLR5PLANEDIR: str = os.path.join(EXTERNAL_DB, f"{plane.name}.txt")
read_polars_3d(XFLR5PLANEDIR, plane.name)

# from ICARUS.visualization.airplane.db_polars import plot_airplane_polars
# planenames = [plane.name]
# plot_airplane_polars(
#     planenames,
#     solvers=["AVL"],
#     plots=[["AoA", "CL"], ["AoA", "CD"], ["AoA", "Cm"]],
#     size=(6, 7),
# )
from ICARUS.computation.solvers.XFLR5.dynamic_analysis import xflr_eigs

eig_file = os.path.join(EXTERNAL_DB, "hermes_eig.txt")
xflr_long, xflr_late = xflr_eigs(eig_file)

avl_dynamic_analysis_implicit(plane=plane, state=state, solver2D=solver2D)
impl_long, impl_late = process_avl_impl_res(plane, state)

# aoa_trim, u_trim = avldyn.trim_conditions(PLANEDIR, plane)
unstick = State(name="Unstick", airplane=plane, environment=EARTH_ISA, u_freestream=UINF)

unstick.add_polar(
    polar=pol_df,
    polar_prefix="AVL",
    is_dimensional=True,
)

### Pertrubations
epsilons = {
    "u": 0.01,
    "w": 0.01,
    "q": 0.001,
    "theta": 0.01,
    "v": 0.01,
    "p": 0.001,
    "r": 0.001,
    "phi": 0.001,
}

# epsilons = None
unstick.add_all_pertrubations("Central", epsilons)
unstick.get_pertrub()

avl_dynamic_analysis_fd(plane, unstick, solver2D)
df = process_avl_fd_res(plane, unstick)

unstick.set_pertrubation_results(df)
unstick.stability_fd()


print(unstick)
fig, ax = unstick.plot_eigenvalues()

x = [ele.real for ele in xflr_late]
y = [ele.imag for ele in xflr_late]
ax[0].scatter(x, y, marker="x", label="XFLR LAT", color="k")

x = [ele.real for ele in xflr_long]
y = [ele.imag for ele in xflr_long]
ax[1].scatter(x, y, marker="o", label="XFLR LONG", color="k")

x = [ele.real for ele in impl_late]
y = [ele.imag for ele in impl_late]
ax[0].scatter(x, y, marker="x", label="IMPL LATE", color="m")

x = [ele.real for ele in impl_long]
y = [ele.imag for ele in impl_long]
ax[1].scatter(x, y, marker="o", label="IMPL LONG", color="m")

ax[0].legend()
ax[1].legend()
import matplotlib.pyplot as plt

plt.show()
