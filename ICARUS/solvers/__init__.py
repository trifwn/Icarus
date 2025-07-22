"""============================================
ICARUS Solvers package
============================================

.. toctree: generated/
    :hidden:

    ICARUS.Solvers.OpenFoam
    ICARUS.Solvers.Xfoil
    ICARUS.Solvers.Foil2Wake
    ICARUS.Solvers.GenuVP
    ICARUS.Solvers.XFLR5

    isort:skip_file
"""

from ICARUS import INSTALL_DIR
from . import XFLR5
from . import Foil2Wake
from . import GenuVP
from . import Icarus_LSPT
from . import OpenFoam
from . import Xfoil

__all__ = [
    # Solver Modules
    "XFLR5",
    "Foil2Wake",
    "GenuVP",
    "Icarus_LSPT",
    "OpenFoam",
    "Xfoil",
]

import os

runOFscript = os.path.join(INSTALL_DIR, "Solvers", "OpenFoam", "runFoam.sh")
setup_of_script = os.path.join(INSTALL_DIR, "Solvers", "OpenFoam", "setupFoam.sh")
logOFscript = os.path.join(INSTALL_DIR, "Solvers", "OpenFoam", "logFoam.sh")
