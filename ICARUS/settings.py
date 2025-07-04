import builtins
import logging
import multiprocessing
import os
import platform
import sys

import jsonpickle.ext.numpy as jsonpickle_numpy
import jsonpickle.ext.pandas as jsonpickle_pd
import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# Detect if running in a Jupyter Notebook environment
try:
    from IPython.core.getipython import get_ipython

    if get_ipython() is not None:
        IN_JUPYTER = True
    else:
        IN_JUPYTER = False
except ImportError:
    IN_JUPYTER = False

ICARUS_THEME = "solarized-dark"

# If running in Jupyter, set the console to use Rich's Jupyter console
if IN_JUPYTER:
    ICARUS_CONSOLE = Console(
        file=sys.stdout,
        force_jupyter=True,
        # theme ="solarized-dark",
        # soft_wrap=True,
    )

else:
    # Rich Console for logging and output
    ICARUS_CONSOLE = Console(
        file=sys.stdout,
        force_terminal=True,
        # theme = "solarized-dark",
        # soft_wrap=True,
        # highlight=False,  # Disable syntax highlighting for better performance in large outputs
    )

# Make console the default stream for print and logging
builtins.print = ICARUS_CONSOLE.print

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(message)s",
    handlers=[
        RichHandler(
            console=ICARUS_CONSOLE,
            rich_tracebacks=True,
            show_level=True,
            show_time=True,
            show_path=True,
        ),
    ],
)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logger = logging.getLogger("ICARUS")

# Nice Traceback
# install(show_locals=True)
install()

# np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

PLATFORM = platform.system()

if PLATFORM == "Windows":
    # Check if context has been set
    try:
        if multiprocessing.get_start_method() != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        print(f"Multiprocessing start method set to '{multiprocessing.get_start_method()}'.")
        pass
elif PLATFORM == "Linux":
    try:
        if multiprocessing.get_start_method() != "forkserver":
            multiprocessing.set_start_method("forkserver", force=True)
    except RuntimeError as e:
        print(f"Multiprocessing start method set to '{multiprocessing.get_start_method()}'. Got error: {e}")
        pass

CPU_COUNT: int = multiprocessing.cpu_count()

if CPU_COUNT > 2:
    if PLATFORM == "Windows":
        CPU_TO_USE: int = CPU_COUNT // 2
    else:
        CPU_TO_USE = CPU_COUNT // 2
else:
    CPU_TO_USE = 1

INSTALL_DIR: str = os.path.dirname(os.path.realpath(__file__))
INSTALL_DIR = os.path.abspath(os.path.join(INSTALL_DIR, os.pardir))

HAS_GPU = False
HAS_JAX = False
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


if PLATFORM == "Windows":
    GenuVP3_exe: str = os.path.join(INSTALL_DIR, "bin", "gnvp3.exe")
    GenuVP7_exe: str = os.path.join(INSTALL_DIR, "bin", "gnvp7.exe")
    F2W_exe: str = os.path.join(INSTALL_DIR, "bin", "f2w.exe")
    Foil_Section_exe: str = os.path.join(INSTALL_DIR, "bin", "foil_section.exe")
    AVL_exe: str = os.path.join(INSTALL_DIR, "bin", "avl.exe")
elif PLATFORM == "Linux":
    GenuVP3_exe = os.path.join(INSTALL_DIR, "bin", "gnvp3")
    GenuVP7_exe = os.path.join(INSTALL_DIR, "bin", "gnvp7")
    F2W_exe = os.path.join(INSTALL_DIR, "bin", "f2w")
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

# Register jsonpickle handlers for numpy and pandas
jsonpickle_pd.register_handlers()
jsonpickle_numpy.register_handlers()

ii64 = np.iinfo(np.int64)
f64 = np.finfo(np.float64)
MAX_INT = ii64.max - 1
MAX_FLOAT = float(f64.max)
