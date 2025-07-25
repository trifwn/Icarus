import multiprocessing
import os
import platform

import jsonpickle.ext.numpy as jsonpickle_numpy
import jsonpickle.ext.pandas as jsonpickle_pd
import numpy as np

PLATFORM = platform.system()

if PLATFORM == "Windows":
    # Check if context has been set
    try:
        if multiprocessing.get_start_method() != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        print(
            f"Multiprocessing start method set to '{multiprocessing.get_start_method()}'.",
        )
        pass
elif PLATFORM == "Linux":
    try:
        if multiprocessing.get_start_method() != "forkserver":
            multiprocessing.set_start_method("forkserver", force=True)
    except RuntimeError as e:
        print(
            f"Multiprocessing start method set to '{multiprocessing.get_start_method()}'. Got error: {e}",
        )
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
    EXE_DIR = os.path.join(INSTALL_DIR, "bin", "Windows")

    GenuVP3_exe: str = os.path.join(EXE_DIR, "gnvp3.exe")
    Foil_exe: str = os.path.join(EXE_DIR, "foil2w.exe")
    AVL_exe: str = os.path.join(EXE_DIR, "avl.exe")
    GenuVP7_exe: str = os.path.join(EXE_DIR, "gnvp7.exe")
elif PLATFORM == "Linux":
    EXE_DIR = os.path.join(INSTALL_DIR, "bin", PLATFORM)

    Foil_exe = os.path.join(EXE_DIR, "foil2w")
    AVL_exe = os.path.join(EXE_DIR, "avl")
    GenuVP3_exe = os.path.join(EXE_DIR, "gnvp3")
    GenuVP7_exe = os.path.join(EXE_DIR, "gnvp7")
elif PLATFORM == "Darwin":
    EXE_DIR = os.path.join(INSTALL_DIR, "bin", "MacOS")

    AVL_exe = os.path.join(EXE_DIR, "avl")
    GenuVP3_exe = os.path.join(EXE_DIR, "gnvp3")
    Foil_exe = os.path.join(EXE_DIR, "foil2w-aarch64")
    GenuVP7_exe = os.path.join(EXE_DIR, "gnvp7")
else:
    raise RuntimeError(f"Unsupported platform: {PLATFORM}")

# Check if the files have execution permission
if not os.access(GenuVP3_exe, os.X_OK):
    os.chmod(GenuVP3_exe, 0o755)

# if not os.access(GenuVP7_exe, os.X_OK):
#     os.chmod(GenuVP7_exe, 0o755)

if not os.access(Foil_exe, os.X_OK):
    os.chmod(Foil_exe, 0o755)

if not os.access(AVL_exe, os.X_OK):
    os.chmod(AVL_exe, 0o755)

# Register jsonpickle handlers for numpy and pandas
jsonpickle_pd.register_handlers()
jsonpickle_numpy.register_handlers()

ii64 = np.iinfo(np.int64)
f64 = np.finfo(np.float64)
MAX_INT = ii64.max - 1
MAX_FLOAT = float(f64.max)
