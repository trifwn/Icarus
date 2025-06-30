import matplotlib.pyplot as plt
import numpy as np
from Planes.hermes import hermes

from ICARUS.database import Database
from ICARUS.environment import EARTH_ISA
from ICARUS.flight_dynamics import State
from ICARUS.solvers.AVL import avl_dynamics_fd
from ICARUS.solvers.AVL import avl_dynamics_implicit
from ICARUS.solvers.AVL import avl_polars
from ICARUS.solvers.AVL import process_avl_dynamics_implicit

# CHANGE THIS TO YOUR DATABASE FOLDER
database_folder = "E:\\Icarus\\Data"
# Load the database
DB = Database(database_folder)
plane = hermes("hermes")

env = EARTH_ISA
UINF = 20
solver2D = "Xfoil"
state = State(name="Unstick", airplane=plane, environment=EARTH_ISA, u_freestream=UINF)

angles = np.linspace(-10, 10, 11)

avl_polars(plane, state, solver2D, angles)
# state.plot_polars()

avl_dynamics_implicit(plane=plane, state=state, solver2D=solver2D)
impl_long, impl_late = process_avl_dynamics_implicit(plane, state)


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
state.add_all_pertrubations("Central", epsilons)
state.get_pertrub()

avl_dynamics_fd(plane, state, solver2D)

print(state)
fig = plt.figure(figsize=(12, 6))
axs = fig.subplots(1, 2)
axs = axs.flatten()

state.plot_eigenvalues(axs=axs)

# x = [ele.real for ele in impl_late]
# y = [ele.imag for ele in impl_late]
# axs[0].scatter(x, y, marker="x", label="Implicit", color="m")

# x = [ele.real for ele in impl_long]
# y = [ele.imag for ele in impl_long]
# axs[1].scatter(x, y, marker="o", label="Implicit", color="m")

axs[0].set_title("Lateral")
axs[1].set_title("Longitudinal")
axs[1].legend()
plt.show()
