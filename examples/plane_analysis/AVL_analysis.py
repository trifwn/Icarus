import os

import matplotlib.pyplot as plt
import numpy as np
from Planes.hermes import hermes

from ICARUS import INSTALL_DIR
from ICARUS.database import Database
from ICARUS.environment import EARTH_ISA
from ICARUS.flight_dynamics import State
from ICARUS.solvers.AVL import AVL
from ICARUS.solvers.AVL import AVLParameters
from ICARUS.solvers.AVL import process_avl_dynamics_implicit

# CHANGE THIS TO YOUR DATABASE FOLDER
database_folder = os.path.join(
    INSTALL_DIR,
    "Data",
)
# Load the database
DB = Database(database_folder)

plane = hermes("hermes")

env = EARTH_ISA
UINF = 20
state = State(name="Unstick", airplane=plane, environment=EARTH_ISA, airspeed=UINF)

angles = np.linspace(-10, 10, 11)

avl = AVL()
avl_parameters = AVLParameters()

avl_parameters.run_invscid = True
avl_parameters.use_avl_control = True

try:
    avl.aseq(plane, state, angles, solver_parameters=avl_parameters)
except Exception as e:
    print(f"Error during AVL analysis: {e}")
# state.plot_polars()

avl.stability_implicit(plane=plane, state=state, solver_parameters=avl_parameters)
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
state.print_pertrubations()

avl.stability(plane, state, solver_parameters=avl_parameters)

fig = plt.figure(figsize=(12, 6))
axs = fig.subplots(1, 2)
axs = axs.flatten()

state.plot_eigenvalues(axs=axs)

real_late = [eigen.real for eigen in impl_late]
real_long = [eigen.real for eigen in impl_long]
imag_late = [eigen.imag for eigen in impl_late]
imag_long = [eigen.imag for eigen in impl_long]
# Add the implicit eigenvalues to the plot
axs[0].scatter(
    real_late,
    imag_late,
    color="black",
    label="Implicit Lateral",
    marker="x",
)
axs[1].scatter(
    real_long,
    imag_long,
    color="black",
    label="Implicit Longitudinal",
    marker="x",
)
axs[0].legend()
axs[1].legend()
plt.show()

state.state_space.lateral.print_matrices()
state.state_space.longitudal.print_matrices()
