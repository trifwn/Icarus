"""ICARUS: A Python package for the analysis, modelling and design of aircraft."""

from . import aerodynamics
from . import airfoils
from . import computation
from . import conceptual
from . import core
from . import database
from . import environment
from . import flight_dynamics
from . import mission
from . import vehicle
from . import visualization

__all__ = [
    "aerodynamics",
    "airfoils",
    "computation",
    "conceptual",
    "core",
    "database",
    "environment",
    "flight_dynamics",
    "mission",
    "vehicle",
    "visualization",
]

__version__ = "0.3.0"

import multiprocessing
import os
import platform

platform_os = platform.system()
CPU_COUNT: int = multiprocessing.cpu_count()

if CPU_COUNT > 2:
    if platform_os == "Windows":
        CPU_TO_USE: int = CPU_COUNT // 2
    else:
        CPU_TO_USE = CPU_COUNT // 2
else:
    CPU_TO_USE = 1

INSTALL_DIR: str = os.path.dirname(os.path.realpath(__file__))
INSTALL_DIR = os.path.abspath(os.path.join(INSTALL_DIR, os.pardir))

try:
    import jax

    # Set precision to 64 bits
    jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True

    # CHECK IF JAX IS USING GPU
    try:
        jax.devices("gpu")
        HAS_GPU = True
    except RuntimeError:
        HAS_GPU = False

except ImportError:
    pass

### SOLVER EXECUTABLES ###

if platform_os == "Windows":
    multiprocessing.set_start_method("spawn")
    GenuVP3_exe: str = os.path.join(INSTALL_DIR, "bin", "gnvp3.exe")
    GenuVP7_exe: str = os.path.join(INSTALL_DIR, "bin", "gnvp7.exe")
    F2W_exe: str = os.path.join(INSTALL_DIR, "bin", "f2w.exe")
    Foil_Section_exe: str = os.path.join(INSTALL_DIR, "bin", "foil_section.exe")
    AVL_exe: str = os.path.join(INSTALL_DIR, "bin", "avl.exe")
elif platform_os == "Linux":
    try:
        multiprocessing.set_start_method("forkserver")
    except RuntimeError:
        pass
    GenuVP3_exe = os.path.join(INSTALL_DIR, "bin", "gnvp3")
    GenuVP7_exe = os.path.join(INSTALL_DIR, "bin", "gnvp7")
    # F2W_exe = os.path.join(APPHOME, "bin", "f2w")
    Foil_Section_exe = os.path.join(INSTALL_DIR, "bin", "foil_section")
    AVL_exe = os.path.join(INSTALL_DIR, "bin", "avl")

    # Check if the files have execution permission
    if not os.access(GenuVP3_exe, os.X_OK):
        os.chmod(GenuVP3_exe, 0o755)

    if not os.access(GenuVP7_exe, os.X_OK):
        os.chmod(GenuVP7_exe, 0o755)

    # if not os.access(F2W_exe, os.X_OK):
    # os.chmod(F2W_exe, 0o755)

    if not os.access(Foil_Section_exe, os.X_OK):
        os.chmod(Foil_Section_exe, 0o755)

    if not os.access(AVL_exe, os.X_OK):
        os.chmod(AVL_exe, 0o755)
