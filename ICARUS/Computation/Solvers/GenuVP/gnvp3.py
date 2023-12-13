from typing import Any

from ICARUS.Computation.Analyses.analysis import Analysis
from ICARUS.Computation.Solvers.GenuVP.analyses.angles import process_gnvp_angles_run_3
from ICARUS.Computation.Solvers.GenuVP.analyses.angles import run_gnvp3_angles
from ICARUS.Computation.Solvers.GenuVP.analyses.angles import run_gnvp3_angles_parallel
from ICARUS.Computation.Solvers.GenuVP.analyses.pertrubations import (
    proccess_pertrubation_res_3,
)
from ICARUS.Computation.Solvers.GenuVP.analyses.pertrubations import (
    run_gnvp3_pertrubation_parallel,
)
from ICARUS.Computation.Solvers.GenuVP.analyses.pertrubations import (
    run_gnvp3_pertrubation_serial,
)
from ICARUS.Computation.Solvers.GenuVP.files.gnvp3_interface import gnvp3_execute
from ICARUS.Computation.Solvers.solver import Solver
from ICARUS.Database import DB
from ICARUS.Environment.definition import Environment
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Vehicle.plane import Airplane


def get_gnvp3() -> Solver:
    gnvp3 = Solver(name="gnvp3", solver_type="3D", fidelity=2)

    # # Define GNVP3 Analyses
    options: dict[str, tuple[str, Any]] = {
        "HOMEDIR": (
            "Home Directory",
            str,
        ),
        "CASEDIR": (
            "Case Directory",
            str,
        ),
    }

    solver_options: dict[str, tuple[Any, str, Any]] = {
        "Split_Symmetric_Bodies": (
            False,
            "Split Symmetric Bodies And Contstruct Them In GNVP3",
            bool,
        ),
        "Use_Grid": (False, "Use the Grid generated in the plane object", bool),
        "Integration_Scheme": (
            2,
            "NMETHT=1 for Euler =2 for Adams Bashford time integrat. scheme",
            int,
        ),
        "Tip_Emmision": (
            0,
            "NEMTIP=0,1. The latter means that tip-emission takes place",
            int,
        ),
        "Tip_Emmision_Begins": (
            0,
            "NTIMET Time at which tip-emission begins",
            int,
        ),
        "Leading_Edge_Separation": (
            0,
            "=0(no action), 1(leading-edge separ. takes place)",
            int,
        ),
        "Leading_Edge_Separation_Begins": (
            0,
            "NTIMEL Time at which leading-edge separ. begins",
            int,
        ),
        "Relaxation_Factor": (
            1,
            "RELAXS relaxation factor for the singularity distributions",
            float,
        ),
        "Pot_Convergence_Tolerence": (
            1e-4,
            "EPSDS convergence tolerance of the potential calculations",
            float,
        ),
        "Movement_Levels": (
            4,
            "NLEVELT number of movements levels",
            int,
        ),
        "Vortex_Particle_Count": (
            1,
            "NNEVP0 Number of vortex particles created within a time step per near-wake element of a thin  wing",
            int,
        ),
        "Vortex_Particle_Relaxation": (
            1.0,
            "RELAXU relaxation factor for the emission velocity",
            float,
        ),
        "Minimum_Width_Parameter": (
            1.0,
            "PARVEC parameter for the minimum width of the near-wake element",
            float,
        ),
        "NEMIS": (1, "1 or 2 UKNOWN", int),
        "Bound_Vorticity_Cutoff": (
            1e-3,
            "EPSFB  Cut-off length for the bound vorticity",
            float,
        ),
        "Wake_Vorticity_Cutoff": (
            1e-3,
            "EPSFW  Cut-off length for the near-wake vorticity",
            float,
        ),
        "Cutoff_Length_Sources": (
            0.003,
            "EPSSR  Cut-off length for source distributions",
            float,
        ),
        "Cutoff_Length_Sources2": (
            0.003,
            "EPSDI  Cut-off length for source distributions",
            float,
        ),
        "Vortex_Cutoff_Length_f": (
            1e-1,
            "EPSVR  Cut-off length for the free vortex particles (final)",
            float,
        ),
        "Vortex_Cutoff_Length_i": (
            1e-1,
            "EPSO   Cut-off length for the free vortex particles (init.) ",
            float,
        ),
        "EPSINT": (
            0.001,
            "EPSINT",
            float,
        ),
        "Particle_Dissipation_Factor": (
            0.0,
            "COEF    Factor for the disipation of particles",
            float,
        ),
        "Upper_Deformation_Rate": (
            0.001,
            "RMETM   Upper bound of the deformation rate",
            float,
        ),
        "Wake_Deformation_Parameter": (
            1,
            "IDEFW   Parameter for the deformation induced by the near wake ",
            float,
        ),
        "REFLEN": (
            1000.0,
            "REFLEN  Length used in VELEF for suppresing far-particle calc.",
            float,
        ),
        "Particle_Subdivision_Parameter": (
            0,
            "IDIVVRP Parameter for the subdivision of particles",
            float,
        ),
        "Subdivision_Length_Scale": (
            1000.0,
            "FLENSC  Length scale for the subdivision of particles",
            float,
        ),
        "Wake_Particle_Merging_Parameter": (
            0,
            "NREWAK  Parameter for merging of particles",
            float,
        ),
        "Particle_Merging_Parameter": (
            0,
            "NMER    Parameter for merging of particles",
            float,
        ),
        "Merging_Starting_Distance": (
            0,
            "XREWAK  X starting distance of merging",
            float,
        ),
        "Merging_Radius": (0, "RADMER  Radius for merging", float),
        "Elasticity_Solver": (0, "IYNELST (1=BEAMDYN,2-ALCYONE,3=GAST)", int),
    }

    rerun: Analysis = Analysis("gnvp3", "rerun", gnvp3_execute, options, solver_options)

    options = {
        "plane": (
            "Plane Object",
            Airplane,
        ),
        "environment": (
            "Environment",
            Environment,
        ),
        "solver2D": (
            "2D Solver",
            str,
        ),
        "maxiter": (
            "Max Iterations",
            int,
        ),
        "timestep": (
            "Timestep = 0.05 * chord / u_inf",
            float,
        ),
        "u_freestream": (
            "Velocity Magnitude",
            float,
        ),
        "angles": (
            "Angles to run",
            list[float],
        ),
    }

    angles_serial: Analysis = Analysis(
        "gnvp3",
        "Angles_Serial",
        run_gnvp3_angles,
        options,
        solver_options,
        unhook=process_gnvp_angles_run_3,
    )

    angles_parallel: Analysis = angles_serial << {
        "name": "Angles_Parallel",
        "execute": run_gnvp3_angles_parallel,
        "unhook": process_gnvp_angles_run_3,
    }
    options = {
        "plane": (
            "Plane Object",
            Airplane,
        ),
        "state": (
            "Dynamic State of the airplane",
            State,
        ),
        "environment": (
            "Environment",
            Environment,
        ),
        "solver2D": (
            "2D Solver",
            str,
        ),
        "maxiter": (
            "Max Iterations",
            int,
        ),
        "timestep": (
            "Timestep",
            float,
        ),
        "u_freestream": (
            "Velocity Magnitude",
            float,
        ),
        "angle": (
            "Angle to run",
            float,
        ),
    }

    pertrubation_serial: Analysis = Analysis(
        "gnvp3",
        "Pertrubation_Serial",
        run_gnvp3_pertrubation_serial,
        options,
        solver_options,
        unhook=proccess_pertrubation_res_3,
    )

    pertrubation_parallel: Analysis = pertrubation_serial << {
        "name": "Pertrubation_Parallel",
        "execute": run_gnvp3_pertrubation_parallel,
        "unhook": proccess_pertrubation_res_3,
    }

    gnvp3.add_analyses(
        [
            rerun,
            angles_serial,
            angles_parallel,
            pertrubation_serial,
            pertrubation_parallel,
        ],
    )

    return gnvp3


# # EXAMPLE USAGE
if __name__ == "__main__":
    from ICARUS.Database.utils import angle_to_case
    import os

    HOMEDIR = os.getcwd()
    gnvp3 = get_gnvp3()
    analysis = gnvp3.available_analyses_names()[0]
    gnvp3.set_analyses(analysis)
    options = gnvp3.get_analysis_options()

    plane = list(DB.vehicles_db.planes.items())[0][1]
    CASEDIR = plane.CASEDIR + "/" + angle_to_case(0.0) + "/"
    options["HOMEDIR"].value = HOMEDIR
    options["CASEDIR"].value = CASEDIR
    # gnvp3.run()
