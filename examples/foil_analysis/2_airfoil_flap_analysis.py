import matplotlib.pyplot as plt
import numpy as np

from ICARUS.computation.solvers.solver import Solver
from ICARUS.computation.solvers.Xfoil.xfoil import Xfoil
from ICARUS.core.types import FloatArray
from ICARUS.database import Database

# CHANGE THIS TO YOUR DATABASE FOLDER
database_folder = "E:\\Icarus\\Data"

# Load the database
DB = Database(database_folder)
airfoil = DB.get_airfoil("NACA0009")

# PARAMETERS FOR ESTIMATION
chord_max: float = 0.5
chord_min: float = 0.1
u_max: float = 35.0
u_min: float = 5.0
viscosity: float = 1.56e-5
speed_of_sound: float = 340.3

# MACH ESTIMATION
mach_max: float = 0.0
# mach_min: float = calc_mach(10, speed_of_sound)
# mach: FloatArray = np.linspace(mach_max, mach_min, 10)
MACH: float = mach_max

# REYNOLDS ESTIMATION
RE_MIN = 8e4
RE_MAX = 1.5e6
NUM_BINS = 12
REYNOLDS_BINS = np.logspace(-2.2, 0, NUM_BINS) * (RE_MAX - RE_MIN) + RE_MIN
reynolds = REYNOLDS_BINS

# ANGLE OF ATTACK SETUP
aoa_min: float = -10
aoa_max: float = 16
num_of_angles: int = int((aoa_max - aoa_min) * 2 + 1)
angles: FloatArray = np.linspace(
    start=aoa_min,
    stop=aoa_max,
    num=num_of_angles,
)

# Transition to turbulent Boundary Layer
ftrip_up: dict[str, float] = {"pos": 0.2, "neg": 0.1}
ftrip_low: dict[str, float] = {"pos": 0.1, "neg": 0.2}
Ncrit = 9


print(f"\nRunning airfoil {airfoil}\n")

for flap_angle in np.arange(-12.5, -30, -2.5):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    airfoil_flap = airfoil.flap(
        flap_hinge_chord_percentage=0.63,
        flap_angle=flap_angle,
        chord_extension=1.0,
    )
    airfoil_flap.repanel_spl(200)

    xfoil: Solver = Xfoil()

    # Import Analysis
    analysis: str = xfoil.get_analyses_names()[1]  # Run
    xfoil.select_analysis(analysis)

    # Get Options
    options = xfoil.get_analysis_options(verbose=False)
    solver_parameters = xfoil.get_solver_parameters()

    # Set Options
    options.airfoil = airfoil_flap
    options.mach = MACH
    options.reynolds = reynolds
    options.min_aoa = aoa_min
    options.max_aoa = aoa_max
    options.aoa_step = 0.5

    # Set Solver Options
    solver_parameters.max_iter = 200
    solver_parameters.Ncrit = 9
    solver_parameters.xtr = (0.2, 0.1)
    solver_parameters.print = False
    solver_parameters.repanel_n = 140

    xfoil.define_analysis(options, solver_parameters)
    xfoil.print_analysis_options()

    # RUN
    xfoil.execute(parallel=False)
    # Get polar
    polar = DB.foils_db.get_polars(airfoil_flap.name.upper())
    polar.data
    print("\n\n\n")
    # fig = polar.plot()
    # fig.show()
    # plt.show(block=True)
