import matplotlib.pyplot as plt
import numpy as np

import ICARUS.Computation.Solvers.AVL.input as avlinp
import ICARUS.Computation.Solvers.AVL.polars as avlpol
from ICARUS.Computation.Solvers.AVL.post import polar_postprocess
from ICARUS.Environment.definition import EARTH_ISA
from ICARUS.Vehicle.lifting_surface import Lifting_Surface
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Vehicle.utils import define_linear_chord
from ICARUS.Vehicle.utils import define_linear_span


mw = Lifting_Surface(
    name="Main_Wing_1",
    airfoil="2412",
    origin=np.array([0.0, 0.0, 0.0]),
    orientation=np.array([0.0, 0.0, 0.0]),
    is_symmetric=True,
    sweep_offset=0.1,
    span=2.0,
    dih_angle=0.0,
    chord_fun=define_linear_chord,
    chord=np.array([0.4, 0.4]),
    span_fun=define_linear_span,
    # twist_fun: Callable[[float, int], FloatArray],
    N=30,
    M=10,
    mass=1.0,
)
el = Lifting_Surface(
    name="Elevator",
    airfoil="0012",
    origin=np.array([1.25, 0, 0]),
    orientation=np.array([0.0, 0.0, 0.0]),
    is_symmetric=True,
    sweep_offset=0.0,
    span=1.0,
    dih_angle=0.0,
    chord_fun=define_linear_chord,
    chord=np.array([0.2, 0.2]),
    span_fun=define_linear_span,
    # twist_fun: Callable[[float, int], FloatArray],
    N=20,
    M=5,
    mass=1.0,
)
rud = Lifting_Surface(
    name="Rudder",
    airfoil="0012",
    origin=np.array([1.3, 0, 0.2]),
    orientation=np.array([0, 0.0, 90.0]),
    is_symmetric=True,
    sweep_offset=0.0,
    span=1.0,
    dih_angle=0.0,
    chord_fun=define_linear_chord,
    chord=np.array([0.1, 0.1]),
    span_fun=define_linear_span,
    # twist_fun: Callable[[float, int], FloatArray],
    N=20,
    M=5,
    mass=1.0,
)
w_polar = np.array([[-1, 0.23, 1.375], [0.034, 0.0064, 0.04]])  # NACA 2412 at Re = 5e5
el_polar = np.array([[-1.15, 0, 1.15], [0.038, 0.00768, 0.038]])  # NACA 0012 at Re = 3e5
rud_polar = el_polar

pl = Airplane("plane_100", [mw, el, rud])


env = EARTH_ISA

pms = [(1.9, [-0.1, 0, 0.2], "payload"), (0.9, [-0.1, 0, 0.2], "pa2")]
pl.add_point_masses(pms)

import os
from ICARUS.Database import DB

PLANEDIR: str = os.path.join(DB.vehicles_db.DATADIR, pl.CASEDIR, "AVL")
os.makedirs(PLANEDIR, exist_ok=True)
avlinp.make_input_files(PLANEDIR, pl, env, 1.0, 1.0, w_polar, 1.0, 1.0, el_polar, 1.0, 1.0, rud_polar)

angles = np.linspace(-10, 10, 11)
avlpol.case_def(PLANEDIR, pl, angles)
avlpol.case_setup(PLANEDIR, pl)
avlpol.case_run(PLANEDIR, pl, angles)

pol_df = polar_postprocess(PLANEDIR, angles)

fig, ax = plt.subplots(1, 1)
ax.plot(pol_df["AOA"], pol_df["CL"], label="CL")
ax.plot(pol_df["AOA"], pol_df["CD"], label="CD")
ax.plot(pol_df["AOA"], pol_df["Cm"], label="Cm")
ax.legend()
plt.show()
