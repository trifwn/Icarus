import os

import numpy as np
from Vehicles.Planes.hermes import hermes

import ICARUS.Computation.Solvers.AVL.dynamics as avldyn
import ICARUS.Computation.Solvers.AVL.input as avlinp
import ICARUS.Computation.Solvers.AVL.polars as avlpol
import ICARUS.Computation.Solvers.AVL.post as avlpst
from ICARUS.Computation.Solvers.AVL.post import process_avl_angle_run
from ICARUS.Computation.Solvers.XFLR5.polars import read_polars_3d
from ICARUS.Database import DB
from ICARUS.Database import EXTERNAL_DB
from ICARUS.Environment.definition import EARTH_ISA
from ICARUS.Flight_Dynamics.state import State


plane = hermes("hermes")

env = EARTH_ISA
UINF = 20

PLANEDIR: str = os.path.join(DB.vehicles_db.DATADIR, plane.directory, "AVL")
angles = np.linspace(-10, 10, 11)

avlinp.make_input_files(PLANEDIR, plane, env, UINF)
avlpol.case_def(PLANEDIR, plane, angles)
avlpol.case_setup(PLANEDIR, plane)
avlpol.case_run(PLANEDIR, plane, angles)
pol_df = process_avl_angle_run(plane, angles)


planenames = [plane.name]
from ICARUS.Database import EXTERNAL_DB

for name in planenames:
    if name.startswith("XFLR"):
        continue
    if f"XFLR_{name}" not in planenames:
        try:
            XFLR5PLANEDIR: str = os.path.join(EXTERNAL_DB, f"{name}.txt")
            read_polars_3d(XFLR5PLANEDIR, name)
            print(f"Imported XFLR polar for {name}")
            planenames.append(f"XFLR_{name}")
        except FileNotFoundError:
            pass

# from ICARUS.Visualization.airplane.db_polars import plot_airplane_polars

# plot_airplane_polars(
#     planenames,
#     solvers=["AVL"],
#     plots=[["AoA", "CL"], ["AoA", "CD"], ["AoA", "Cm"]],
#     size=(6, 7),
# )
impl_long, impl_late = avldyn.implicit_eigs(
    PLANEDIR,
    plane,
    EARTH_ISA,
    UINF=UINF,
)


# aoa_trim, u_trim = avldyn.trim_conditions(PLANEDIR, plane)
unstick = State(
    name="Unstick",
    airplane=plane,
    environment=EARTH_ISA,
    polar=pol_df,
    polar_prefix="AVL",
    is_dimensional=False,
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

avldyn.finite_difs(
    plane=plane,
    state=unstick,
)

df = avlpst.finite_difs_post(plane, unstick)
from ICARUS.Computation.Solvers.XFLR5.dynamic_analysis import xflr_eigs

eig_file = os.path.join(EXTERNAL_DB, "hermes_eig.txt")
xflr_long, xflr_late = xflr_eigs(eig_file)

unstick.set_pertrubation_results(df)
unstick.stability_fd()
unstick.eigenvalue_analysis()
print(unstick)
fig, ax = unstick.plot_eigenvalues(plot_lateral=False)

# x = [ele.real for ele in xflr_late]
# y = [ele.imag for ele in xflr_late]
# ax.scatter(x, y, marker="x", label="XFLR LAT", color="k")

x = [ele.real for ele in xflr_long]
y = [ele.imag for ele in xflr_long]
ax.scatter(x, y, marker="o", label="XFLR LONG", color="k")

# x = [ele.real for ele in impl_late]
# y = [ele.imag for ele in impl_late]
# ax.scatter(x, y, marker="x", label="IMPL LATE", color="m")

x = [ele.real for ele in impl_long]
y = [ele.imag for ele in impl_long]
ax.scatter(x, y, marker="o", label="IMPL LONG", color="m")

ax.legend()
import matplotlib.pyplot as plt

print(xflr_late)
plt.show()
